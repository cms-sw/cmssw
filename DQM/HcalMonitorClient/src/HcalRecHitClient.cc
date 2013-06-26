#include "DQM/HcalMonitorClient/interface/HcalRecHitClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <iostream>

/*
 * \file HcalRecHitClient.cc
 * 
 * $Date: 2012/06/18 08:23:10 $
 * $Revision: 1.54 $
 * \author J. Temple
 * \brief Dead Cell Client class
 */

HcalRecHitClient::HcalRecHitClient(std::string myname)
{
  name_=myname;
}

HcalRecHitClient::HcalRecHitClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("RecHitFolder","RecHitMonitor_Hcal/"); // RecHitMonitor_Hcal  
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("RecHit_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("RecHit_BadChannelStatusMask",
                                                          ps.getUntrackedParameter<int>("BadChannelStatusMask",
											0)); // identify channel status values to mask

  minerrorrate_ = ps.getUntrackedParameter<double>("RecHit_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.01));
  minevents_    = ps.getUntrackedParameter<int>("RecHit_minevents",
						ps.getUntrackedParameter<int>("minevents",1));
  enoughevents_=false;
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  ProblemCells=0;
  ProblemCellsByDepth=0;
}

void HcalRecHitClient::analyze()
{
  if (debug_>2) std::cout <<"\tHcalRecHitClient::analyze()"<<std::endl;
 
  TH2F* OccupancyByDepth[4];
  TH2F* SumEnergyByDepth[4];
  TH2F* SumTimeByDepth[4];
  TH2F* SqrtSumEnergy2ByDepth[4];

  TH2F* OccupancyThreshByDepth[4];
  TH2F* SumEnergyThreshByDepth[4];
  TH2F* SumTimeThreshByDepth[4];
  TH2F* SqrtSumEnergy2ThreshByDepth[4];

  std::vector<std::string> name = HcalEtaPhiHistNames();

  MonitorElement* me;
  bool gotHistos=true;

  for (int i=0;i<4;++i)
    {
      std::string s=subdir_+"Distributions_AllRecHits/"+name[i]+"RecHit Occupancy";
      me=dqmStore_->get(s.c_str());
      if (me==0) {if (debug_>0) std::cout <<"Could not get histogram "<<s<<std::endl; gotHistos=false; break;}
      OccupancyByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, OccupancyByDepth[i], debug_);
      s=subdir_+"Distributions_AllRecHits/sumplots/"+name[i]+"RecHit Summed Energy GeV";
      me=dqmStore_->get(s.c_str());
      if (me==0) {if (debug_>0) std::cout <<"Could not get histogram "<<s<<std::endl; gotHistos=false; break;}
      SumEnergyByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, SumEnergyByDepth[i], debug_);
      s=subdir_+"Distributions_AllRecHits/sumplots/"+name[i]+"RecHit Summed Time nS";
      me=dqmStore_->get(s.c_str());
      if (me==0) {if (debug_>0) std::cout <<"Could not get histogram "<<s<<std::endl; gotHistos=false; break;}
      SumTimeByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, SumTimeByDepth[i], debug_);
      s=subdir_+"Distributions_AllRecHits/sumplots/"+name[i]+"RecHit Sqrt Summed Energy2 GeV";
      me=dqmStore_->get(s.c_str());
      if (me==0) {if (debug_>0) std::cout <<"Could not get histogram "<<s<<std::endl; gotHistos=false; break;}
      SqrtSumEnergy2ByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, SqrtSumEnergy2ByDepth[i], debug_);

      // Threshold histograms
      s=subdir_+"Distributions_PassedMinBias/"+name[i]+"Above Threshold RecHit Occupancy";
      me=dqmStore_->get(s.c_str());
      if (me==0) {if (debug_>0) std::cout <<"Could not get histogram "<<s<<std::endl; gotHistos=false; break;}
      OccupancyThreshByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, OccupancyThreshByDepth[i], debug_);
      s=subdir_+"Distributions_PassedMinBias/sumplots/"+name[i]+"Above Threshold RecHit Summed Energy GeV";
      me=dqmStore_->get(s.c_str());
      if (me==0) {if (debug_>0) std::cout <<"Could not get histogram "<<s<<std::endl; gotHistos=false; break;}
      SumEnergyThreshByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, SumEnergyThreshByDepth[i], debug_);
      s=subdir_+"Distributions_PassedMinBias/sumplots/"+name[i]+"Above Threshold RecHit Summed Time nS";
      me=dqmStore_->get(s.c_str());
      if (me==0) {if (debug_>0) std::cout <<"Could not get histogram "<<s<<std::endl; gotHistos=false; break;}
      SumTimeThreshByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, SumTimeThreshByDepth[i], debug_);
      s=subdir_+"Distributions_PassedMinBias/sumplots/"+name[i]+"Above Threshold RecHit Sqrt Summed Energy2 GeV";
      me=dqmStore_->get(s.c_str());
      if (me==0) {if (debug_>0) std::cout <<"Could not get histogram "<<s<<std::endl; gotHistos=false; break;}
      SqrtSumEnergy2ThreshByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, SqrtSumEnergy2ThreshByDepth[i], debug_);
    }
  if (gotHistos==false)
    {
      if (debug_>0) std::cout <<"<HcalRecHitClient::calculateProblems()> Not all histograms could be found; skipping normalization"<<std::endl;
      return;
    }

  // Clear histograms before re-filling
  meHBEnergy_1D->Reset();
  meHBEnergyRMS_1D->Reset();
  meHEEnergy_1D->Reset();
  meHEEnergyRMS_1D->Reset();
  meHOEnergy_1D->Reset();
  meHOEnergyRMS_1D->Reset();
  meHFEnergy_1D->Reset();
  meHFEnergyRMS_1D->Reset();
  meHBEnergyThresh_1D->Reset();
  meHBEnergyRMSThresh_1D->Reset();
  meHEEnergyThresh_1D->Reset();
  meHEEnergyRMSThresh_1D->Reset();
  meHOEnergyThresh_1D->Reset();
  meHOEnergyRMSThresh_1D->Reset();
  meHFEnergyThresh_1D->Reset();
  meHFEnergyRMSThresh_1D->Reset();

  for (int mydepth=0;mydepth<4;++mydepth)
    {
      for (int eta=0;eta<OccupancyByDepth[mydepth]->GetNbinsX();++eta)
	{
	  // eta+1=1:  ieta = -42
	  // eta+1=13: ieta = -29

	  for (int phi=0;phi<72;++phi)
	    {
	      if (OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1)>0)
		{
		  // fill 1D plots
		  if (isHB(eta,mydepth+1)) 
		    {
		      if (validDetId(HcalBarrel, CalcIeta(HcalBarrel, eta, mydepth+1), phi+1, mydepth+1))
			{
			  meHBEnergy_1D->Fill(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
			  meHBEnergyRMS_1D->Fill(sqrt(pow(SqrtSumEnergy2ByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1)-pow(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)));
			}
		    } 
		  else if (isHE(eta,mydepth+1)) 
		    {
		      if (validDetId(HcalEndcap, CalcIeta(HcalEndcap, eta, mydepth+1), phi+1, mydepth+1)) 
			{
			  
			  meHEEnergy_1D->Fill(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
			  meHEEnergyRMS_1D->Fill(sqrt(pow(SqrtSumEnergy2ByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1)-pow(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)));
			}
		    }
		  else if (isHO(eta,mydepth+1)) 
		    {
		      if (validDetId(HcalOuter, CalcIeta(HcalOuter, eta, mydepth+1), phi+1, mydepth+1)) 
			{
			  meHOEnergy_1D->Fill(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
			  meHOEnergyRMS_1D->Fill(sqrt(pow(SqrtSumEnergy2ByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1)-pow(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)));
			}
		    } 
		  else if (isHF(eta,mydepth+1)) 
		    {
		      if (validDetId(HcalForward, CalcIeta(HcalForward, eta, mydepth+1), phi+1, mydepth+1)) 
			{
			  meHFEnergy_1D->Fill(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
			  meHFEnergyRMS_1D->Fill(sqrt(pow(SqrtSumEnergy2ByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1)-pow(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)));
			}
		    }
		  // normalize 2D plots by number of events
		 
		  meEnergyByDepth->depth[mydepth]->setBinContent(eta+1, phi+1, SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
		  meTimeByDepth->depth[mydepth]->setBinContent(eta+1, phi+1, SumTimeByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
		} // (OccupancyByDepth[mydepth]->GetBinContent(eta+1,phi+1)>0)
                
	      if (OccupancyThreshByDepth[mydepth]==0) continue;
	      if (OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)>0)
		{
		  // fill 1D plots
		  if (isHB(eta,mydepth+1)) 
		    {
		      if (validDetId(HcalBarrel, CalcIeta(HcalBarrel, eta, mydepth+1), phi+1, mydepth+1))
			{
			  meHBEnergyThresh_1D->Fill(SumEnergyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1));
			  double RMS=pow(SqrtSumEnergy2ThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1,phi+1)-pow(SumEnergyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1),2);
			  RMS=pow(fabs(RMS),0.5);
			  meHBEnergyRMSThresh_1D->Fill(RMS);
			}
		    } 
		  else if (isHE(eta,mydepth+1)) 
		    {
		      if (validDetId(HcalEndcap, CalcIeta(HcalEndcap, eta, mydepth+1), phi+1, mydepth+1)) 
			{
			  
			  meHEEnergyThresh_1D->Fill(SumEnergyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1));
			  double RMS=pow(SqrtSumEnergy2ThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1,phi+1)-pow(SumEnergyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1),2);
			  RMS=pow(fabs(RMS),0.5);
			  meHEEnergyRMSThresh_1D->Fill(RMS);
			}
		    }
		  else if (isHO(eta,mydepth+1)) 
		    {
		      if (validDetId(HcalOuter, CalcIeta(HcalOuter, eta, mydepth+1), phi+1, mydepth+1)) 
			{
			  meHOEnergyThresh_1D->Fill(SumEnergyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1));
			  double RMS=pow(SqrtSumEnergy2ThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1,phi+1)-pow(SumEnergyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1),2);
			  RMS=pow(fabs(RMS),0.5);
			  meHOEnergyRMSThresh_1D->Fill(RMS);
			}
		    } 
		  else if (isHF(eta,mydepth+1)) 
		    {
		      if (validDetId(HcalForward, CalcIeta(HcalForward, eta, mydepth+1), phi+1, mydepth+1)) 
			{
			  meHFEnergyThresh_1D->Fill(SumEnergyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1));
			  double RMS=pow(SqrtSumEnergy2ThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1,phi+1)-pow(SumEnergyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1),2);
			  RMS=pow(fabs(RMS),0.5);
			  meHFEnergyRMSThresh_1D->Fill(RMS);		  
			}
		    }
		  // fill 2D plots
		  meEnergyThreshByDepth->depth[mydepth]->setBinContent(eta+1, phi+1, SumEnergyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1));
		  meTimeThreshByDepth->depth[mydepth]->setBinContent(eta+1, phi+1, SumTimeThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1));
		}
	    } // for (int phi=0;phi<72;++phi)
	} // for (int eta=0;eta<OccupancyByDepth->..;++eta)
    } // for (int mydepth=0;...)

  FillUnphysicalHEHFBins(*meEnergyByDepth);
  FillUnphysicalHEHFBins(*meTimeByDepth);
  FillUnphysicalHEHFBins(*meEnergyThreshByDepth);
  FillUnphysicalHEHFBins(*meTimeThreshByDepth);

  calculateProblems();
}

void HcalRecHitClient::calculateProblems()
{
 if (debug_>2) std::cout <<"\t\tHcalRecHitClient::calculateProblems()"<<std::endl;
  if(!dqmStore_) return;
  double totalevents=0;
  int etabins=0, phibins=0, zside=0;
  double problemvalue=0;
  std::vector<std::string> name = HcalEtaPhiHistNames(); // use this to get EtaPhiHistograms that feed problem calculation, once they exist (see analyze function for example usage of HcalEtaPhiHistNames())

  // Get histograms used in determining rechit problem rate, 
  // and get totalevents from their underflow bins.
  // No such problem histograms are defined so far.
  
  enoughevents_=true;

  if (totalevents==0)
    return;

  // Clear away old problems
  if (ProblemCells!=0)
    {
      ProblemCells->Reset();
      (ProblemCells->getTH2F())->SetMaximum(1.05);
      (ProblemCells->getTH2F())->SetMinimum(0.);
    }
  for  (unsigned int d=0;d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]!=0) 
	{
	  ProblemCellsByDepth->depth[d]->Reset();
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetMaximum(1.05);
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetMinimum(0.);
	}
    }
  
  for (unsigned int d=0;d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]==0) continue;

      // Get total events from some future histogram
      //totalevents=DigiPresentByDepth[d]->GetBinContent(0); // get totalevents from each depth, in case they differ
      if (totalevents==0 || totalevents<minevents_) continue;
      etabins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsX();
      phibins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsY();
      problemvalue=0;
      for (int eta=0;eta<etabins;++eta)
	{
	  int ieta=CalcIeta(eta,d+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      problemvalue=0;
	      // Add histograms here when rechit testing decided upon
	      /*if (DigiPresentByDepth[d]!=0 && DigiPresentByDepth[d]->GetBinContent(eta+1,phi+1)>0) 
		problemvalue=totalevents; */
	      
	      if (problemvalue==0) continue;
	      problemvalue/=totalevents; // problem value is a rate; should be between 0 and 1
	      problemvalue = std::min(1.,problemvalue);
	      
	      zside=0;
	      if (isHF(eta,d+1)) // shift ieta by 1 for HF
		ieta<0 ? zside = -1 : zside = 1;
	      
	      // For problem cells that exceed our allowed rate,
	      // set the values to 999 if the cells are already marked in the status database
	      if (problemvalue>minerrorrate_)
		{
		  HcalSubdetector subdet=HcalEmpty;
		  if (isHB(eta,d+1))subdet=HcalBarrel;
		  else if (isHE(eta,d+1)) subdet=HcalEndcap;
		  else if (isHF(eta,d+1)) subdet=HcalForward;
		  else if (isHO(eta,d+1)) subdet=HcalOuter;
		  HcalDetId hcalid(subdet, ieta, phi+1, (int)(d+1));
		  if (badstatusmap.find(hcalid)!=badstatusmap.end())
		    problemvalue=999;
		}
	      
	      
	      ProblemCellsByDepth->depth[d]->setBinContent(eta+1,phi+1,problemvalue);
	      if (ProblemCells!=0) ProblemCells->Fill(ieta+zside,phi+1,problemvalue);
	    } // loop on phi
	} // loop on eta
    } // loop on depth

  if (ProblemCells==0)
    {
      if (debug_>0) std::cout <<"<HcalRecHitClient::analyze> ProblemCells histogram does not exist!"<<std::endl;
      return;
    }

  // Normalization of ProblemCell plot, in the case where there are errors in multiple depths
  etabins=(ProblemCells->getTH2F())->GetNbinsX();
  phibins=(ProblemCells->getTH2F())->GetNbinsY();
  for (int eta=0;eta<etabins;++eta)
    {
      for (int phi=0;phi<phibins;++phi)
	{
	  if (ProblemCells->getBinContent(eta+1,phi+1)>1. && ProblemCells->getBinContent(eta+1,phi+1)<999)
	    ProblemCells->setBinContent(eta+1,phi+1,1.);
	}
    }
  FillUnphysicalHEHFBins(*ProblemCellsByDepth);
  FillUnphysicalHEHFBins(ProblemCells);
  return;

}

void HcalRecHitClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalRecHitClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}
void HcalRecHitClient::endJob(){}

void HcalRecHitClient::beginRun(void)
{
  if (debug_>1)  std::cout <<"<HcalRecHitClient::endRun>"<<std::endl;

  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalRecHitClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  dqmStore_->setCurrentFolder(subdir_);
  problemnames_.clear();
  ProblemCells=dqmStore_->book2D(" ProblemRecHits",
				 "Problem RecHit Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_rechits");
  ProblemCellsByDepth=new EtaPhiHists();
  ProblemCellsByDepth->setup(dqmStore_," Problem RecHit Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());

  nevts_=0;

  dqmStore_->setCurrentFolder(subdir_+"Distributions_AllRecHits");
  meEnergyByDepth = new EtaPhiHists();
  meEnergyByDepth->setup(dqmStore_,"RecHit Average Energy","GeV");
  meTimeByDepth = new EtaPhiHists();
  meTimeByDepth->setup(dqmStore_,"RecHit Average Time","nS");
  // set all average times to -1000 by default (so that they don't show up on plots
  for (unsigned int i=0;i<meTimeByDepth->depth.size();++i)
    {
      (meTimeByDepth->depth[i]->getTH2F())->SetMinimum(-150);
      (meTimeByDepth->depth[i]->getTH2F())->SetMaximum(150);
      int etabins=(meTimeByDepth->depth[i]->getTH2F())->GetNbinsX();
      int phibins=(meTimeByDepth->depth[i]->getTH2F())->GetNbinsY();
      for (int x=1;x<=etabins;++x)
	for (int y=1;y<=phibins;++y)
	  meTimeByDepth->depth[i]->setBinContent(x,y,-1000);
    }

  dqmStore_->setCurrentFolder(subdir_+"Distributions_PassedMinBias");
  meEnergyThreshByDepth = new EtaPhiHists();
  meEnergyThreshByDepth->setup(dqmStore_,"Above Threshold RecHit Average Energy","GeV");
  meTimeThreshByDepth = new EtaPhiHists();
  meTimeThreshByDepth->setup(dqmStore_,"Above Threshold RecHit Average Time","nS");
 // set all average times to -1000 by default (so that they don't show up on plots
  for (unsigned int i=0;i<meTimeThreshByDepth->depth.size();++i)
    {
      (meTimeThreshByDepth->depth[i]->getTH2F())->SetMinimum(-150);
      (meTimeThreshByDepth->depth[i]->getTH2F())->SetMaximum(150);
      int etabins=(meTimeThreshByDepth->depth[i]->getTH2F())->GetNbinsX();
      int phibins=(meTimeThreshByDepth->depth[i]->getTH2F())->GetNbinsY();
      for (int x=1;x<=etabins;++x)
	for (int y=1;y<=phibins;++y)
	  meTimeThreshByDepth->depth[i]->setBinContent(x,y,-1000);
    }

  dqmStore_->setCurrentFolder(subdir_+"Distributions_AllRecHits/rechit_1D_plots/");
  meHBEnergy_1D=dqmStore_->book1D("HB_energy_1D","HB Average Energy Per RecHit;Energy (GeV)",200,-5,15);
  meHEEnergy_1D=dqmStore_->book1D("HE_energy_1D","HE Average Energy Per RecHit;Energy (GeV)",200,-5,15);
  meHOEnergy_1D=dqmStore_->book1D("HO_energy_1D","HO Average Energy Per RecHit;Energy (GeV)",200,-10,20);
  meHFEnergy_1D=dqmStore_->book1D("HF_energy_1D","HF Average Energy Per RecHit;Energy (GeV)",200,-5,15);

  meHBEnergyRMS_1D=dqmStore_->book1D("HB_energy_RMS_1D","HB Energy RMS Per RecHit;Energy (GeV)",250,0,5);
  meHEEnergyRMS_1D=dqmStore_->book1D("HE_energy_RMS_1D","HE Energy RMS Per RecHit;Energy (GeV)",250,0,5);
  meHOEnergyRMS_1D=dqmStore_->book1D("HO_energy_RMS_1D","HO Energy RMS Per RecHit;Energy (GeV)",250,0,5);
  meHFEnergyRMS_1D=dqmStore_->book1D("HF_energy_RMS_1D","HF Energy RMS Per RecHit;Energy (GeV)",250,0,5);

  dqmStore_->setCurrentFolder(subdir_+"Distributions_PassedMinBias/rechit_1D_plots/");
  meHBEnergyThresh_1D=dqmStore_->book1D("HB_energyThresh_1D","HB Average Energy Per RecHit Above Threshold;Energy (GeV)",200,-5,35);
  meHEEnergyThresh_1D=dqmStore_->book1D("HE_energyThresh_1D","HE Average Energy Per RecHit Above Threshold;Energy (GeV)",200,-5,35);
  meHOEnergyThresh_1D=dqmStore_->book1D("HO_energyThresh_1D","HO Average Energy Per RecHit Above Threshold;Energy (GeV)",300,-10,50);
  meHFEnergyThresh_1D=dqmStore_->book1D("HF_energyThresh_1D","HF Average Energy Per RecHit Above Threshold;Energy (GeV)",200,-5,95);

  meHBEnergyRMSThresh_1D=dqmStore_->book1D("HB_energy_RMSThresh_1D","HB Energy RMS Per RecHit Above Threshold;Energy (GeV)",200,0,10);
  meHEEnergyRMSThresh_1D=dqmStore_->book1D("HE_energy_RMSThresh_1D","HE Energy RMS Per RecHit Above Threshold;Energy (GeV)",200,0,10);
  meHOEnergyRMSThresh_1D=dqmStore_->book1D("HO_energy_RMSThresh_1D","HO Energy RMS Per RecHit Above Threshold;Energy (GeV)",200,0,10);
  meHFEnergyRMSThresh_1D=dqmStore_->book1D("HF_energy_RMSThresh_1D","HF Energy RMS Per RecHit Above Threshold;Energy (GeV)",200,0,20);
}

void HcalRecHitClient::endRun(void)
{
  if (debug_>1)  std::cout <<"<HcalRecHitClient::endRun>"<<std::endl;
  analyze();
}

void HcalRecHitClient::setup(void){}
void HcalRecHitClient::cleanup(void){}

bool HcalRecHitClient::hasErrors_Temp(void)
{
  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalRecHitClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
      return false;
    }
  int problemcount=0;
  int ieta=-9999;

  for (int depth=0;depth<4; ++depth)
    {
      int etabins  = (ProblemCells->getTH2F())->GetNbinsX();
      int phibins  = (ProblemCells->getTH2F())->GetNbinsY();
      for (int hist_eta=0;hist_eta<etabins;++hist_eta)
        {
          for (int hist_phi=0; hist_phi<phibins;++hist_phi)
            {
              ieta=CalcIeta(hist_eta,depth+1);
	      if (ieta==-9999) continue;
	      if (ProblemCellsByDepth->depth[depth]==0)
		continue;
	      if (ProblemCellsByDepth->depth[depth]->getBinContent(hist_eta,hist_phi)>minerrorrate_)
		++problemcount;
	    } // for (int hist_phi=1;...)
	} // for (int hist_eta=1;...)
    } // for (int depth=0;...)

  if (problemcount>0) return true;
  return false;
}

bool HcalRecHitClient::hasWarnings_Temp(void){return false;}
bool HcalRecHitClient::hasOther_Temp(void){return false;}
bool HcalRecHitClient::test_enabled(void){return true;}



void HcalRecHitClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // rechit quality not used to update channel status yet; see dead cell client for example



} //void HcalRecHitClient::updateChannelStatus

HcalRecHitClient::~HcalRecHitClient()
{}
