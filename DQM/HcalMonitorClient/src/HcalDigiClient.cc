#include "DQM/HcalMonitorClient/interface/HcalDigiClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <iostream>

/*
 * \file HcalDigiClient.cc
 * 
 * $Date: 2012/06/18 08:23:10 $
 * $Revision: 1.70 $
 * \author J. Temple
 * \brief DigiClient class
 */

HcalDigiClient::HcalDigiClient(std::string myname)
{
  name_=myname;
}

HcalDigiClient::HcalDigiClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("DigiFolder","DigiMonitor_Hcal/"); // DigiMonitor_Hcal  
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("Digi_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("Digi_BadChannelStatusMask",
                                                          ps.getUntrackedParameter<int>("BadChannelStatusMask",
											(1<<HcalChannelStatus::HcalCellDead))); // identify channel status values to mask
  
  minerrorrate_ = ps.getUntrackedParameter<double>("Digi_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.05));
  minevents_    = ps.getUntrackedParameter<int>("Digi_minevents",
						ps.getUntrackedParameter<int>("minevents",1));
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  ProblemCellsByDepth=0;
  ProblemCells=0;

  HFTiming_averageTime=0;
}

void HcalDigiClient::analyze()
{
  if (debug_>2) std::cout <<"\tHcalDigiClient::analyze()"<<std::endl;
  calculateProblems();

  // Get Pawel's timing plots to form averages
  TH2F* TimingStudyTime=0;
  TH2F* TimingStudyOcc=0;
  std::string s=subdir_+"HFTimingStudy/sumplots/HFTiming_Total_Time";
  
  MonitorElement* me=dqmStore_->get(s.c_str());
  if (me!=0)
    TimingStudyTime=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_,TimingStudyTime, debug_);

  s=subdir_+"HFTimingStudy/sumplots/HFTiming_Occupancy";
  me=dqmStore_->get(s.c_str());
  if (me!=0)
    TimingStudyOcc=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_,TimingStudyOcc, debug_);

  if (HFTiming_averageTime!=0)
    {
      HFTiming_averageTime->Reset();
      if (TimingStudyTime!=0 && TimingStudyOcc!=0)
        {
          int etabins=(HFTiming_averageTime->getTH2F())->GetNbinsX();
          int phibins=(HFTiming_averageTime->getTH2F())->GetNbinsY();
          for (int x=1;x<=etabins;++x)
	    for (int y=1;y<=phibins;++y)
	      if (TimingStudyOcc->GetBinContent(x,y)!=0)
	        HFTiming_averageTime->setBinContent(x,y,TimingStudyTime->GetBinContent(x,y)*1./TimingStudyOcc->GetBinContent(x,y));
        }
      HFTiming_averageTime->getTH2F()->SetMinimum(0);
    }
}

void HcalDigiClient::calculateProblems()
{
 if (debug_>2) std::cout <<"\t\tHcalDigiClient::calculateProblems()"<<std::endl;
  if(!dqmStore_) return;
  int totalevents=0;
  int etabins=0, phibins=0, zside=0;
  double problemvalue=0;

  // Clear away old problems
  if (ProblemCells!=0)
    {
      ProblemCells->Reset();
      (ProblemCells->getTH2F())->SetMaximum(1.05);
      (ProblemCells->getTH2F())->SetMinimum(0.);
    }
  for  (unsigned int d=0;ProblemCellsByDepth!=0 && d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]!=0) 
	{
	  ProblemCellsByDepth->depth[d]->Reset();
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetMaximum(1.05);
	  (ProblemCellsByDepth->depth[d]->getTH2F())->SetMinimum(0.);
	}
    }

  // Get histograms that are used in testing
  TH2F* BadDigisByDepth[4];
  TH2F* GoodDigisByDepth[4];

  std::vector<std::string> name = HcalEtaPhiHistNames();

  bool gothistos=true;

  MonitorElement* me;
  for (int i=0;i<4;++i)
    {
      std::string s=subdir_+"bad_digis/bad_digi_occupancy/"+name[i]+"Bad Digi Map";
      me=dqmStore_->get(s.c_str());
      if (me==0) 
	{
	  gothistos=false;
	  if (debug_>0) std::cout <<"<HcalDigiClient::calculateProblems> Could not get histogram with name "<<s<<std::endl;
	}
      BadDigisByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, BadDigisByDepth[i], debug_);

      s=subdir_+"good_digis/digi_occupancy/"+name[i]+" Digi Eta-Phi Occupancy Map";
      me=dqmStore_->get(s.c_str());
      if (me==0) 
	{
	  gothistos=false;
	  if (debug_>0) std::cout <<"<HcalDigiClient::calculateProblems> Could not get histogram with name "<<s<<std::endl;
	}
      GoodDigisByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, BadDigisByDepth[i], debug_);
    }

  if (gothistos==false)
    {
      if (debug_>0) std::cout <<"<HcalDigiClient::calculateProblems> Unable to get all necessary histograms to evaluate problem rate"<<std::endl;
      return;
    }

  for (unsigned int d=0;ProblemCellsByDepth!=0 && d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]==0) continue;

      if (BadDigisByDepth[d]==0 || GoodDigisByDepth[d]==0) continue;
      totalevents=(int)GoodDigisByDepth[d]->GetBinContent(0,0);
      if (totalevents<minevents_ ) continue;
      enoughevents_=true;
      etabins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsX();
      phibins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  int ieta=CalcIeta(eta,d+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      problemvalue=0; // problem fraction sums over all three tests
	      if (BadDigisByDepth[d]->GetBinContent(eta+1,phi+1) > 0) // bad cells found
		problemvalue=(BadDigisByDepth[d]->GetBinContent(eta+1,phi+1)*1./(BadDigisByDepth[d]->GetBinContent(eta+1,phi+1)+GoodDigisByDepth[d]->GetBinContent(eta+1,phi+1)));
	      
	      zside=0;
	      if (isHF(eta,d+1)) // shift ieta by 1 for HF
		ieta<0 ? zside = -1 : zside = 1;
	      
	      // For problem cells that exceed our allowed rate,
	      // set the values to -1 if the cells are already marked in the status database
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
      if (debug_>0) std::cout <<"<HcalDigiClient::analyze> ProblemCells histogram does not exist!"<<std::endl;
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


void HcalDigiClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalDigiClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}

void HcalDigiClient::endJob(){}

void HcalDigiClient::beginRun(void)
{
  enoughevents_=false;
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalDigiClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  dqmStore_->setCurrentFolder(subdir_);
  problemnames_.clear();
  ProblemCells=dqmStore_->book2D(" ProblemDigis",
				 " Problem Digi Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_digis");
  ProblemCellsByDepth = new EtaPhiHists();
  ProblemCellsByDepth->setup(dqmStore_," Problem Digi Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());

  nevts_=0;

  dqmStore_->setCurrentFolder(subdir_+"HFTimingStudy");
  HFTiming_averageTime=dqmStore_->book2D("HFTimingStudy_Average_Time","HFTimingStudy Average Time (time sample)",83,-41.5,41.5,72,0.5,72.5);
}

void HcalDigiClient::endRun(void){analyze();}

void HcalDigiClient::setup(void){}
void HcalDigiClient::cleanup(void){}

bool HcalDigiClient::hasErrors_Temp(void)
{
  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalDigiClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
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

bool HcalDigiClient::hasWarnings_Temp(void){return false;}
bool HcalDigiClient::hasOther_Temp(void){return false;}
bool HcalDigiClient::test_enabled(void){return true;}


void HcalDigiClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // digi client does not alter channel status yet;
  // look at dead cell or hot cell clients for example code
} //void HcalDigiClient::updateChannelStatus

HcalDigiClient::~HcalDigiClient()
{}
