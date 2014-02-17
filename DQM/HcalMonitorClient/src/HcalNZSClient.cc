#include "DQM/HcalMonitorClient/interface/HcalNZSClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <iostream>

/*
 * \file HcalNZSClient.cc
 * 
 * $Date: 2012/06/18 08:23:10 $
 * $Revision: 1.5 $
 * \author J. Temple
 * \brief Hcal NZS Client class
 */

HcalNZSClient::HcalNZSClient(std::string myname)
{
  name_=myname;
}

HcalNZSClient::HcalNZSClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("NZSFolder","NZSMonitor_Hcal/"); // NZSMonitor_Hcal/
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("NZS_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("NZS_BadChannelStatusMask",
							  ps.getUntrackedParameter<int>("BadChannelStatusMask",0));
  
  minerrorrate_ = ps.getUntrackedParameter<double>("NZS_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.01));
  minevents_    = ps.getUntrackedParameter<int>("NZS_minevents",
						ps.getUntrackedParameter<int>("minevents",1));
  ProblemCells=0;
  ProblemCellsByDepth=0;
}

void HcalNZSClient::analyze()
{
  if (debug_>2) std::cout <<"\tHcalNZSClient::analyze()"<<std::endl;
  calculateProblems();
}

void HcalNZSClient::calculateProblems()
{
 if (debug_>2) std::cout <<"\t\tHcalNZSClient::calculateProblems()"<<std::endl;
  if(!dqmStore_) return;
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
  enoughevents_=true;
  // Get histograms that are used in testing
  // currently none used,

  std::vector<std::string> name = HcalEtaPhiHistNames();

  /*
    // This is a sample of how to get a histogram from the task that can then be used for evaluation purposes
  TH2F* DigiPresentByDepth[i]
  MonitorElement* me;
  for (int i=0;i<4;++i)
    {
      std::string s=subdir_+"dead_digi_never_present/"+name[i]+"Digi Present At Least Once";
      me=dqmStore_->get(s.c_str());
      DigiPresentByDepth[i]=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_, DigiPresentByDepth[i], debug_);
    }      
  */


  // Because we're clearing and re-forming the problem cell histogram here, we don't need to do any cute
  // setting of the underflow bin to 0, and we can plot results as a raw rate between 0-1.
  
  for (unsigned int d=0;ProblemCellsByDepth!=0 && d<ProblemCellsByDepth->depth.size();++d)
    {
      if (ProblemCellsByDepth->depth[d]==0) continue;
    
      //totalevents=DigiPresentByDepth[d]->GetBinContent(0);
      totalevents=0;
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
	      //if (DigiPresentByDepth[d]!=0 && DigiPresentByDepth[d]->GetBinContent(eta+1,phi+1)==0) problemvalue=totalevents;
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
      if (debug_>0) std::cout <<"<HcalNZSClient::analyze> ProblemCells histogram does not exist!"<<std::endl;
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

void HcalNZSClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalNZSClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}
void HcalNZSClient::endJob(){}

void HcalNZSClient::beginRun(void)
{
  enoughevents_=false;
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalNZSClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  dqmStore_->setCurrentFolder(subdir_);
  problemnames_.clear();

  // Put the appropriate name of your problem summary here
  ProblemCells=dqmStore_->book2D(" ProblemNZS",
				 " Problem NZS Rate for all HCAL;ieta;iphi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_NZS");
  ProblemCellsByDepth = new EtaPhiHists();
  ProblemCellsByDepth->setup(dqmStore_," Problem NZS Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());
  nevts_=0;
}

void HcalNZSClient::endRun(void){analyze();}

void HcalNZSClient::setup(void){}
void HcalNZSClient::cleanup(void){}

bool HcalNZSClient::hasErrors_Temp(void)
{
  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalNZSClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
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

bool HcalNZSClient::hasWarnings_Temp(void){return false;}
bool HcalNZSClient::hasOther_Temp(void){return false;}
bool HcalNZSClient::test_enabled(void){return true;}


void HcalNZSClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // This gets called by HcalMonitorClient
  // trigger primitives don't yet contribute to channel status (though they could...)
  // see dead or hot cell code for an example

} //void HcalNZSClient::updateChannelStatus

HcalNZSClient::~HcalNZSClient()
{}
