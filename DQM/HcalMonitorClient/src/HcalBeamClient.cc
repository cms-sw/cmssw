#include "DQM/HcalMonitorClient/interface/HcalBeamClient.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include <iostream>

/*
 * \file HcalBeamClient.cc
 * 
 * $Date: 2012/06/18 08:23:10 $
 * $Revision: 1.20 $
 * \author J. Temple
 * \brief Hcal Beam Monitor Client class
 */

HcalBeamClient::HcalBeamClient(std::string myname)
{
  name_=myname;
}

HcalBeamClient::HcalBeamClient(std::string myname, const edm::ParameterSet& ps)
{
  name_=myname;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("BeamFolder","BeamMonitor_Hcal/"); // BeamMonitor_Hcal
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("Beam_validHtmlOutput",true);
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  // known hot/dead channels blacked out on plot
  badChannelStatusMask_   = ps.getUntrackedParameter<int>("Beam_BadChannelStatusMask",
							  ps.getUntrackedParameter<int>("BadChannelStatusMask", 
											((1<<HcalChannelStatus::HcalCellDead)||
											 (1<<HcalChannelStatus::HcalCellHot))
											));
							  
  minerrorrate_ = ps.getUntrackedParameter<double>("Beam_minerrorrate",
						   ps.getUntrackedParameter<double>("minerrorrate",0.05));
  minevents_ = ps.getUntrackedParameter<int>("Beam_minLS",1);
  // minevents_    = ps.getUntrackedParameter<int>("Beam_minevents",
  // 						ps.getUntrackedParameter<int>("minevents",1));
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  ProblemCells=0;
  ProblemCellsByDepth=0;
}

void HcalBeamClient::analyze()
{
  if (debug_>2) std::cout <<"\tHcalBeamClient::analyze()"<<std::endl;
  enoughevents_=false;
  calculateProblems();
}

void HcalBeamClient::calculateProblems()
{
  if (debug_>2) std::cout <<"\t\tHcalBeamClient::calculateProblems()"<<std::endl;
  if(!dqmStore_) return;
  double totalLumiBlocks=0;
  // reminder:: lumi histograms work a bit differently, counting total number of lumi blocks, not total number of events
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
  // currently none used,

  // get the dead and hot cell lumi histograms
  TH2F* dead = 0;
  TH2F* hot  = 0;
  MonitorElement* me;
  me=dqmStore_->get(subdir_+"Lumi/HFlumi_total_deadcells");
  if (me!=0)
    dead=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_,dead,debug_);
  else if (debug_>0) std::cout <<" <HcalBeamClient::calculateProblems> Unable to get dead cell plot 'HFlumi_total_deadcells"<<std::endl;
  me=dqmStore_->get(subdir_+"Lumi/HFlumi_total_hotcells");
  if (me!=0)
    hot=HcalUtilsClient::getHisto<TH2F*>(me, cloneME_,dead,debug_);
  else if (debug_>0) std::cout <<" <HcalBeamClient::calculateProblems> Unable to get hot cell plot 'HFlumi_total_hotcells"<<std::endl;
  int myieta=0;
  int mydepth=0;
  int myiphi=0;

  enoughevents_=true; // beam client works a little differently, counting only lumi blocks that have enough events to process.  For this reason, let client continue running regardless of lumi blocks processed

  if (dead!=0 || hot!=0)
    {
      if (dead!=0) 
	{
	  totalLumiBlocks=dead->GetBinContent(-1,-1);
	  etabins=dead->GetNbinsX();
	  phibins=dead->GetNbinsY();
	}
      else 
	{
	  totalLumiBlocks=hot->GetBinContent(-1,-1);
	  etabins=hot->GetNbinsX();
	  phibins=hot->GetNbinsY();
	}
      if (totalLumiBlocks<minevents_ || totalLumiBlocks==0)
	return;
      for (int i=0;i<etabins;++i)
	{
	  i<=3 ? myieta = i-36 : myieta=i+29; // separate HFM, HFP
	  if (abs(myieta)==33 || abs(myieta)==34)
	    mydepth=1;
	  else if (abs(myieta)==35 || abs(myieta)==36)
	    mydepth=2;
	  for (int j=0;j<phibins;++j)
	    {
	      problemvalue=0;
	      myiphi=2*j+1; // lumi HF histograms only have 36 bins
	      if (dead!=0 && dead->GetBinContent(i+1,j+1)*1./totalLumiBlocks>minerrorrate_)
		{
		  problemvalue+=dead->GetBinContent(i+1,j+1)*1./totalLumiBlocks;
		  if (debug_>1) std::cout <<"<HcalBeamClient::calculateProblem>  Dead cell found at ieta = "<<myieta<<" iphi = "<<myiphi<<"  depth = "<<mydepth<<std::endl;
		}
	      if (hot!=0 &&  hot->GetBinContent(i+1,j+1)*1./totalLumiBlocks>minerrorrate_)
		{
		  problemvalue+=hot->GetBinContent(i+1,j+1)*1./totalLumiBlocks;
		  if (debug_>1) std::cout <<"<HcalBeamClient::calculateProblem>  hot cell found at ieta = "<<myieta<<" iphi = "<<myiphi<<"  depth = "<<mydepth<<std::endl;
		}
	      if (problemvalue==0) continue;

	      // Search for known bad problems in channel status db
	      HcalDetId hcalid(HcalForward, myieta, myiphi, mydepth);
	      if (badstatusmap.find(hcalid)!=badstatusmap.end())
		problemvalue=999; 	
	      myieta<0 ?  zside=-1 : zside=1;
	      ProblemCellsByDepth->depth[mydepth-1]->Fill(myieta+zside,myiphi,problemvalue);
	      if (ProblemCells!=0) ProblemCells->Fill(myieta+zside,myiphi,problemvalue);
	    }
	}
    }


  if (ProblemCells==0)
    {
      if (debug_>0) std::cout <<"<HcalBeamClient::analyze> ProblemCells histogram does not exist!"<<std::endl;
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

void HcalBeamClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalBeamClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}
void HcalBeamClient::endJob(){}

void HcalBeamClient::beginRun(void)
{
  enoughevents_=false;
  if (!dqmStore_) 
    {
      if (debug_>0) std::cout <<"<HcalBeamClient::beginRun> dqmStore does not exist!"<<std::endl;
      return;
    }
  dqmStore_->setCurrentFolder(subdir_);
  problemnames_.clear();

  // Put the appropriate name of your problem summary here
  if (ProblemCells==0)
    ProblemCells=dqmStore_->book2D(" Problem BeamMonitor",
				   " Problem Beam Monitor Rate for all HCAL;ieta;iphi",
				   85,-42.5,42.5,
				   72,0.5,72.5);
  problemnames_.push_back(ProblemCells->getName());
  if (debug_>1)
    std::cout << "Tried to create ProblemCells Monitor Element in directory "<<subdir_<<"  \t  Failed?  "<<(ProblemCells==0)<<std::endl;
  dqmStore_->setCurrentFolder(subdir_+"problem_beammonitor");
  nevts_=0;
  if (ProblemCellsByDepth!=0) return; // histograms already set up
  ProblemCellsByDepth = new EtaPhiHists();
  ProblemCellsByDepth->setup(dqmStore_," Problem BeamMonitor Rate");
  for (unsigned int i=0; i<ProblemCellsByDepth->depth.size();++i)
    problemnames_.push_back(ProblemCellsByDepth->depth[i]->getName());
}

void HcalBeamClient::endRun(void){analyze();}

void HcalBeamClient::setup(void){}
void HcalBeamClient::cleanup(void){}

bool HcalBeamClient::hasErrors_Temp(void)
{
  if (!ProblemCells)
    {
      if (debug_>1) std::cout <<"<HcalBeamClient::hasErrors_Temp>  ProblemCells histogram does not exist!"<<std::endl;
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

bool HcalBeamClient::hasWarnings_Temp(void){return false;}
bool HcalBeamClient::hasOther_Temp(void){return false;}
bool HcalBeamClient::test_enabled(void){return true;}


void HcalBeamClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  // This gets called by HcalMonitorClient
  // trigger primitives don't yet contribute to channel status (though they could...)
  // see dead or hot cell code for an example

} //void HcalBeamClient::updateChannelStatus

HcalBeamClient::~HcalBeamClient()
{}
