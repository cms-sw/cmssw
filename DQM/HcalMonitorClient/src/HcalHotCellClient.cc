#include "DQM/HcalMonitorClient/interface/HcalHotCellClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <iostream>

/*
 * \file HcalHotCellClient.cc
 * 
 * \author J. Temple
 * \brief Hot Cell Client class
 */
 
HcalHotCellClient::HcalHotCellClient(std::string myname)
{
  name_=myname;
}

HcalHotCellClient::HcalHotCellClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("HotCellFolder","HotCellMonitor_Hcal/"); // HotCellMonitor_Hcal  
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("HotCell_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("HotCell_BadChannelStatusMask",
  							  ps.getUntrackedParameter<int>("BadChannelStatusMask",
  											(1<<HcalChannelStatus::HcalCellHot))); // identify channel status values to mask
  // badChannelStatusMask_   = ps.getUntrackedParameter<int>("HotCell_BadChannelStatusMask", (1<<1)); // identify channel status values to mask

  minerrorrate_ = ps.getUntrackedParameter<double>("HotCell_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.25));
  minevents_    = ps.getUntrackedParameter<int>("HotCell_minevents",
						ps.getUntrackedParameter<int>("minevents",100));
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  ProblemCellsByDepth=0;
  ProblemCells=0;
  
  doProblemCellSetup_ = true;
}

void HcalHotCellClient::analyze(DQMStore::IBooker &ib, DQMStore::IGetter &ig)
{
  if (debug_>2) std::cout <<"\tHcalHotCellClient::analyze()"<<std::endl;
  if ( doProblemCellSetup_ ) setupProblemCells(ib,ig);
  calculateProblems(ib,ig);
}

void HcalHotCellClient::calculateProblems(DQMStore::IBooker &ib, DQMStore::IGetter &ig)
{
  if (debug_>2) std::cout <<"\t\tHcalHotCellClient::calculateProblems()"<<std::endl;
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
  TH2F* HotAboveThresholdByDepth[4];
  TH2F* HotAlwaysAboveThresholdByDepth[4];
  TH2F* HotAboveETThresholdByDepth[4];
  TH2F* HotAlwaysAboveETThresholdByDepth[4];
  TH2F* HotNeighborsByDepth[4];

  std::vector<std::string> name = HcalEtaPhiHistNames();

  bool neighbortest=false;

  MonitorElement* me;
  for (int i=0;i<4;++i)
    {
      // Assume histograms aren't found by default
      HotAboveThresholdByDepth[i]=0;
      HotAlwaysAboveThresholdByDepth[i]=0;
      HotAboveETThresholdByDepth[i]=0;
      HotAlwaysAboveETThresholdByDepth[i]=0;
      HotNeighborsByDepth[i]=0;

      std::string s=subdir_+"hot_rechit_above_threshold/"+name[i]+"Hot Cells Above ET Threshold";
      me=ig.get(s.c_str());
      if (me!=0)HotAboveETThresholdByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HotAboveETThresholdByDepth[i], debug_);

      s=subdir_+"hot_rechit_always_above_threshold/"+name[i]+"Hot Cells Persistently Above ET Threshold";
      me=ig.get(s.c_str());
      if (me!=0)HotAlwaysAboveETThresholdByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HotAlwaysAboveETThresholdByDepth[i], debug_);

      s=subdir_+"hot_rechit_above_threshold/"+name[i]+"Hot Cells Above Energy Threshold";
      me=ig.get(s.c_str());
      if (me!=0)HotAboveThresholdByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HotAboveThresholdByDepth[i], debug_);

      s=subdir_+"hot_rechit_always_above_threshold/"+name[i]+"Hot Cells Persistently Above Energy Threshold";
      me=ig.get(s.c_str());
      if (me!=0)HotAlwaysAboveThresholdByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HotAlwaysAboveThresholdByDepth[i], debug_);

      s=subdir_+"hot_neighbortest/"+name[i]+"Hot Cells Failing Neighbor Test";
      me=ig.get(s.c_str());
      if (me!=0)HotNeighborsByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, HotNeighborsByDepth[i], debug_);
      s=subdir_+"hot_neighbortest/NeighborTestEnabled";
      me=ig.get(s.c_str());
      if (me!=0 && me->getIntValue()==1)
	neighbortest=true;
    }


  // Because we're clearing and re-forming the problem cell histogram here, we don't need to do any cute
  // setting of the underflow bin to 0, and we can plot results as a raw rate between 0-1.

  for (unsigned int d=0;ProblemCellsByDepth!=0 && d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]==0) continue;

      if (HotAboveETThresholdByDepth[d]) totalevents = std::max(totalevents, HotAboveETThresholdByDepth[d]->GetBinContent(0));
      else if (HotAlwaysAboveETThresholdByDepth[d]) totalevents = std::max(totalevents, HotAlwaysAboveETThresholdByDepth[d]->GetBinContent(0));
      else if (HotAboveThresholdByDepth[d]) totalevents = std::max(totalevents, HotAboveThresholdByDepth[d]->GetBinContent(0));
      else if (HotAlwaysAboveThresholdByDepth[d]) totalevents = std::max(totalevents, HotAlwaysAboveThresholdByDepth[d]->GetBinContent(0));
      else if (neighbortest==true && HotNeighborsByDepth[d]) totalevents = std::max(totalevents, HotNeighborsByDepth[d]->GetBinContent(0));
      else if (debug_>0) std::cout <<"<HcalHotCellClient::calculateProblems> No evaluation histograms found; no valid hot tests enabled?" << std::endl;
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
	      problemvalue=0; // problem fraction sums over all three tests
	      // If cell is never-present in all runs, then problemvalue = event
	      if (HotAboveETThresholdByDepth[d]!=0)
		problemvalue+=HotAboveETThresholdByDepth[d]->GetBinContent(eta+1,phi+1);
	      if (HotAboveThresholdByDepth[d]!=0)
		problemvalue+=HotAboveThresholdByDepth[d]->GetBinContent(eta+1,phi+1);
	      if (HotAlwaysAboveThresholdByDepth[d]!=0)
		problemvalue+=HotAlwaysAboveThresholdByDepth[d]->GetBinContent(eta+1,phi+1);
	      if (neighbortest==true && HotNeighborsByDepth[d]!=0)
		problemvalue+=HotNeighborsByDepth[d]->GetBinContent(eta+1,phi+1);
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
      if (debug_>0) std::cout <<"<HcalHotCellClient::analyze> ProblemCells histogram does not exist!"<<std::endl;
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



void HcalHotCellClient::endJob(){}

void HcalHotCellClient::setupProblemCells(DQMStore::IBooker &ib, DQMStore::IGetter & ig)
{

  ib.setCurrentFolder(subdir_);
  problemnames_.clear();
  ProblemCells=ib.book2D(" ProblemHotCells",
				 " Problem Hot Cell Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  ib.setCurrentFolder(subdir_+"problem_hotcells");
  ProblemCellsByDepth=new EtaPhiHists();
  ProblemCellsByDepth->setup(ib," Problem Hot Cell Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());

  doProblemCellSetup_ = false;

}

void HcalHotCellClient::beginRun(void)
{
  enoughevents_=false;
  nevts_=0;
}

//void HcalHotCellClient::endRun(void){analyze();}

void HcalHotCellClient::setup(void){}
void HcalHotCellClient::cleanup(void){}

bool HcalHotCellClient::hasErrors_Temp(void)
{
  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalHotCellClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
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

bool HcalHotCellClient::hasWarnings_Temp(void){return false;}
bool HcalHotCellClient::hasOther_Temp(void){return false;}
bool HcalHotCellClient::test_enabled(void){return true;}


void HcalHotCellClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // This gets called by HcalMonitorClient
  if (nevts_<minevents_) return; // not enough events to make judgment; don't create new status file

  float binval;
  int ieta=0;
  int iphi=0;
  int etabins=0;
  int phibins=0;
  
  int subdet=0;
  if (debug_>1)
    {
      std::cout <<"<HcalHotCellClient>  Summary of Hot Cells in Run: "<<std::endl;
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
	      if (!(topo_->validDetId((HcalSubdetector)(subdet), ieta, iphi, d+1))) continue;
	      
	      int hotcell=0;
	      if (binval>minerrorrate_)
		hotcell=1;
	      if (hotcell==1 && debug_>0)
		std::cout <<"Hot Cell :  subdetector = "<<subdet<<" (eta,phi,depth) = ("<<ieta<<", "<<iphi<<", "<<d+1<<"):  "<<binval*100.<<"%"<<std::endl;
	      
	      // DetID not found in quality list; add it.  (This shouldn't happen!)
	      if (myqual.find(myid)==myqual.end())
		{
		  myqual[myid]=(hotcell<<HcalChannelStatus::HcalCellHot);  // 
		}
	      else
		{
		  int mask=(1<<HcalChannelStatus::HcalCellHot);
		  // hot cell found; 'or' the hot cell mask with existing ID
		  if (hotcell==1)
		    myqual[myid] |=mask;
		  // cell is not found, 'and' the inverse of the mask with the existing ID.
		  // Does this work correctly?  I think so, but need to verify.
		  // Also, do we want to allow the client to turn off hot cell masks, or only add them?
		  else
		    myqual[myid] &=~mask;
		}
	    } // for (int hist_phi=1;hist_phi<=phibins;++hist_phi)
	} // for (int hist_eta=1;hist_eta<=etabins;++hist_eta)
    } // for (int d=0;d<4;++d)


} //void HcalHotCellClient::updateChannelStatus

HcalHotCellClient::~HcalHotCellClient()
{
  if ( ProblemCellsByDepth ) delete ProblemCellsByDepth;
}
