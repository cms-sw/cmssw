#include "DQM/HcalMonitorClient/interface/HcalDeadCellClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <iostream>

/*
 * \file HcalDeadCellClient.cc
 * 
 * $Date: 2012/06/18 08:23:10 $
 * $Revision: 1.76 $
 * \author J. Temple
 * \brief Dead Cell Client class
 */

HcalDeadCellClient::HcalDeadCellClient(std::string myname)
{
  name_=myname;
  std::cout <<"MY NAME = "<<name_.c_str()<<std::endl;
}

HcalDeadCellClient::HcalDeadCellClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("DeadCellFolder","DeadCellMonitor_Hcal/"); // DeadCellMonitor_Hcal  
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("DeadCell_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("DeadCell_BadChannelStatusMask",
							  ps.getUntrackedParameter<int>("BadChannelStatusMask",
											(1<<HcalChannelStatus::HcalCellDead)));  // identify channel status values to mask
  
  minerrorrate_ = ps.getUntrackedParameter<double>("DeadCell_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.25));
  minevents_    = ps.getUntrackedParameter<int>("DeadCell_minevents",
						ps.getUntrackedParameter<int>("minevents",1000));

  excludeHOring2_backup_=ps.getUntrackedParameter<bool>("excludeHOring2_backup",false);  // this is used only if excludeHOring2 value from Dead Cell task can't be read
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  ProblemCellsByDepth=0;
  ProblemCells=0;

}

void HcalDeadCellClient::analyze()
{
  if (debug_>2) std::cout <<"\tHcalDeadCellClient::analyze()"<<std::endl;
  calculateProblems();
}

void HcalDeadCellClient::calculateProblems()
{


  if (debug_>2) std::cout <<"\t\tHcalDeadCellClient::calculateProblems()"<<std::endl;
  if(!dqmStore_) 
    {
      if (debug_>2) std::cout <<"DQM STORE DOESN'T EXIST"<<std::endl;
      return;
    }

  MonitorElement* temp_present;

  temp_present=dqmStore_->get(subdir_+"ExcludeHOring2");
  int excludeFromHOring2 = 0;
  if (temp_present)
    {
      excludeFromHOring2 = temp_present->getIntValue();
      if (debug_>2) 
	std::cout <<"Read 'excludeFromHOring2' from HcalMonitorTask output; value = "<<excludeFromHOring2<<std::endl;
    }
  else
    {
      excludeHOring2_backup_==true ?  excludeFromHOring2=1 : excludeFromHOring2=0;
      if (debug_>2) 
	std::cout <<"Could not read excludeFromHOring2 from HcalMonitorTasks; using value from cfg file:  "<<excludeFromHOring2<<std::endl;
    }
  
  // Don't fill histograms if nothing from Hcal is present
  if (HBpresent_!=1)
    {
      temp_present=dqmStore_->get(prefixME_+"HcalInfo/HBpresent");
      if (temp_present!=0)
        HBpresent_=temp_present->getIntValue();
    }
  if (HEpresent_!=1)
    {
      temp_present=dqmStore_->get(prefixME_+"HcalInfo/HEpresent");
      if (temp_present!=0)
        HEpresent_=temp_present->getIntValue();
    }
  if (HOpresent_!=1)
    {
      temp_present=dqmStore_->get(prefixME_+"HcalInfo/HOpresent");
      if (temp_present!=0)
        HOpresent_=temp_present->getIntValue();
    }
  if (HFpresent_!=1)
    {
      temp_present=dqmStore_->get(prefixME_+"HcalInfo/HFpresent");
      if (temp_present!=0)
        HFpresent_=temp_present->getIntValue();
    }
  // Never saw any data from any Hcal FED; don't count as dead.
  if (HBpresent_==-1 && HEpresent_==-1 && HOpresent_==-1 && HFpresent_==-1)
    return;
  double totalevents=0;
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
  TH2F* DigiPresentByDepth[4];
  TH2F* RecentMissingDigisByDepth[4];
  TH2F* RecHitsPresentByDepth[4];
  TH2F* RecentMissingRecHitsByDepth[4];

  std::vector<std::string> name = HcalEtaPhiHistNames();

  MonitorElement* me;
  for (int i=0;i<4;++i)
    {
      // Assume that histograms can't be found
      DigiPresentByDepth[i]=0;
      RecentMissingDigisByDepth[i]=0;
      RecHitsPresentByDepth[i]=0;
      RecentMissingRecHitsByDepth[i]=0;
      
      std::string s=subdir_+"dead_digi_never_present/"+name[i]+"Digi Present At Least Once";
      me=dqmStore_->get(s.c_str());
      if (me!=0) DigiPresentByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, DigiPresentByDepth[i], debug_);
      
      s=subdir_+"dead_digi_often_missing/"+name[i]+"Dead Cells with No Digis";
      me=dqmStore_->get(s.c_str());
      if (me!=0) RecentMissingDigisByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, RecentMissingDigisByDepth[i], debug_);
     
      s=subdir_+"dead_rechit_never_present/"+name[i]+"RecHit Above Threshold At Least Once";
      me=dqmStore_->get(s.c_str());
      if (me!=0) RecHitsPresentByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, RecHitsPresentByDepth[i], debug_);

       s=subdir_+"dead_rechit_often_missing/"+name[i]+"RecHits Failing Energy Threshold Test";
      me=dqmStore_->get(s.c_str());
      if (me!=0)RecentMissingRecHitsByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, RecentMissingRecHitsByDepth[i], debug_);

    }

  // Because we're clearing and re-forming the problem cell histogram here, we don't need to do any cute
  // setting of the underflow bin to 0, and we can plot results as a raw rate between 0-1.
  
  for (unsigned int d=0;ProblemCellsByDepth!=0 && d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]==0) continue;

      if (DigiPresentByDepth[d]==0) continue;
      // Get number of entries from DigiPresent histogram 
      // (need to do this for offline DQM combinations of output)
      totalevents=DigiPresentByDepth[d]->GetBinContent(0);
      if (totalevents==0 || totalevents<minevents_) continue;
      enoughevents_=true; // kind of a hack here
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
	      // Don't count problems in HO ring 2 if the "excludeFromHOring2" bit is in use
	      if (isHO(eta,d+1) && excludeFromHOring2>0 && isSiPM(ieta,phi+1,d+1)==false && abs(ieta)>10)
		continue;

	      // Never-present histogram is a boolean, with underflow bin = 1 (for each instance)
	      // Offline DQM adds up never-present histograms from multiple outputs
	      // For now, we want offline DQM to show LS-based 'occupancies', rather than simple boolean on/off
	      // May change in the future?
	      
	      // If cell is never-present in all runs, then problemvalue = event
	      if (DigiPresentByDepth[d]!=0 && DigiPresentByDepth[d]->GetBinContent(eta+1,phi+1)==0) 
		problemvalue=totalevents;

	      // Rec Hit presence test
	      else if (RecHitsPresentByDepth[d]!=0)
		{
		  if (RecHitsPresentByDepth[d]->GetBinContent(eta+1,phi+1)==0)
		    problemvalue=totalevents;
		  else if (RecHitsPresentByDepth[d]->GetBinContent(eta+1,phi+1)>1)
		    RecHitsPresentByDepth[d]->SetBinContent(eta+1,phi+1,1);
		}
	      else
		{
		  if (RecentMissingDigisByDepth[d]!=0)
		    problemvalue+=RecentMissingDigisByDepth[d]->GetBinContent(eta+1,phi+1);
		  if (RecentMissingRecHitsByDepth[d]!=0)
		    problemvalue+=RecentMissingRecHitsByDepth[d]->GetBinContent(eta+1,phi+1);
		}
	      
	      if (problemvalue==0) continue;
	      	      	      
	      problemvalue/=totalevents; // problem value is a rate; should be between 0 and 1
	      problemvalue = std::min(1.,problemvalue);
	      
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
      if (debug_>0) std::cout <<"<HcalDeadCellClient::analyze> ProblemCells histogram does not exist!"<<std::endl;
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

void HcalDeadCellClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalDeadCellClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}
void HcalDeadCellClient::endJob(){}

void HcalDeadCellClient::beginRun(void)
{
  enoughevents_=false;
  HBpresent_=-1;
  HEpresent_=-1;
  HOpresent_=-1;
  HFpresent_=-1;
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalDeadCellClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  dqmStore_->setCurrentFolder(subdir_);
  problemnames_.clear();
  ProblemCells=dqmStore_->book2D(" ProblemDeadCells",
				 " Problem Dead Cell Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_deadcells");
  ProblemCellsByDepth = new EtaPhiHists();
  ProblemCellsByDepth->setup(dqmStore_," Problem Dead Cell Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());

  nevts_=0;
}

void HcalDeadCellClient::endRun(void){analyze();}

void HcalDeadCellClient::setup(void){}
void HcalDeadCellClient::cleanup(void){}

bool HcalDeadCellClient::hasErrors_Temp(void)
{
  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalDeadCellClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
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

bool HcalDeadCellClient::hasWarnings_Temp(void){return false;}
bool HcalDeadCellClient::hasOther_Temp(void){return false;}
bool HcalDeadCellClient::test_enabled(void){return true;}



void HcalDeadCellClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // This gets called by HcalMonitorClient

  if (enoughevents_==false) return; // not enough events to make judgment; don't create new status file

  float binval;
  int ieta=0;
  int iphi=0;
  int etabins=0;
  int phibins=0;
  
  int subdet=0;
  if (debug_>1)
    {
      std::cout <<"<HcalDeadCellClient>  Summary of Dead Cells in Run: "<<std::endl;
      std::cout <<"(Error rate must be >= "<<minerrorrate_*100.<<"% )"<<std::endl;  
    }
  for (int d=0;d<4;++d)
    {
      etabins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsX();
      phibins=(ProblemCellsByDepth->depth[d]->getTH2F())->GetNbinsY();
      for (int hist_eta=0;hist_eta<etabins;++hist_eta)
	{
	  ieta=CalcIeta(hist_eta,d+1);
	  if (ieta==-9999) continue;
	  for (int hist_phi=0;hist_phi<phibins;++hist_phi)
	    {
	      iphi=hist_phi+1;
	      
	      // ProblemCells have already been normalized
	      binval=ProblemCellsByDepth->depth[d]->getBinContent(hist_eta+1,hist_phi+1);
	      
	      // Set subdetector labels for output
	      if (d<2)
		{
		  if (isHB(hist_eta,d+1)) 
		    subdet=HcalBarrel;
		  else if (isHE(hist_eta,d+1)) 
		    subdet=HcalEndcap;
		  else if (isHF(hist_eta,d+1)) 
		    subdet=HcalForward;
		}
	      else if (d==2) 
		subdet=HcalEndcap;
	      else if (d==3) 
		subdet=HcalOuter;
	      // Set correct depth label
	      
	      HcalDetId myid((HcalSubdetector)(subdet), ieta, iphi, d+1);
	      // Need this to keep from flagging non-existent HE/HF cells
	      if (!validDetId((HcalSubdetector)(subdet), ieta, iphi, d+1))
		continue;
	      
	      int deadcell=0;
	      if (binval>minerrorrate_)
		deadcell=1;
	      if (deadcell==1 && debug_>0)
		std::cout <<"Dead Cell :  subdetector = "<<subdet<<" (eta,phi,depth) = ("<<ieta<<", "<<iphi<<", "<<d+1<<"):  "<<binval*100.<<"%"<<std::endl;
	      
	      // DetID not found in quality list; add it.  
	      if (myqual.find(myid)==myqual.end())
		myqual[myid]=(deadcell<<HcalChannelStatus::HcalCellDead);  // deadcell shifted to bit 6
	      else
		{
		  int mask=(1<<HcalChannelStatus::HcalCellDead);
		  // dead cell found; 'or' the dead cell mask with existing ID
		  if (deadcell==1)
		    myqual[myid] |=mask;
		  // cell is not found, 'and' the inverse of the mask with the existing ID.
		  // Does this work correctly?  I think so, but need to verify.
		  // Also, do we want to allow the client to turn off dead cell masks, or only add them?
		  else
		    myqual[myid] &=~mask;
		}
	    } // for (int hist_phi=1;hist_phi<=phibins;++hist_phi)
	} // for (int hist_eta=1;hist_eta<=etabins;++hist_eta)
    } // for (int d=0;d<4;++d)
  
  
} //void HcalDeadCellClient::updateChannelStatus

HcalDeadCellClient::~HcalDeadCellClient()
{}
