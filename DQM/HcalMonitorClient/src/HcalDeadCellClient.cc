#include "DQM/HcalMonitorClient/interface/HcalDeadCellClient.h"
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <math.h>
#include <iostream>
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"

HcalDeadCellClient::HcalDeadCellClient(){} // constructor 

void HcalDeadCellClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  // Get variable values from cfg file
  // Set which dead cell checks will looked at
  deadclient_test_digis_              = ps.getUntrackedParameter<bool>("DeadCellClient_test_digis",true);
  deadclient_test_rechits_            = ps.getUntrackedParameter<bool>("DeadCellClient_test_rechits",true);

  deadclient_checkNevents_ = ps.getUntrackedParameter<int>("DeadCellClient_checkNevents",1000);

  minErrorFlag_ = ps.getUntrackedParameter<double>("DeadCellClient_minErrorFlag",0.0);

  deadclient_makeDiagnostics_ = ps.getUntrackedParameter<bool>("DeadCellClient_makeDiagnosticPlots",false);

  dump2database_ = ps.getUntrackedParameter<bool>("dump2database",false);

  subdets_.push_back("HBE HE HF Depth 1 ");
  subdets_.push_back("HB HE HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO Depth 4 ");


  // Set histograms to NULL
  ProblemCells=0;
  for (int i=0;i<4;++i)
    {
      // Set each array's pointers to NULL
      DigiPresentByDepth[i]=0;
      RecHitsPresentByDepth[i]=0;
      if (deadclient_test_digis_) RecentMissingDigisByDepth[i]=0;
      if (deadclient_test_rechits_) 
	{
	  RecentMissingRecHitsByDepth[i]=0;
	  RecHitsPresentByDepth[i]=0;
	}  
    }
  subdets_.push_back("HBE HE HF Depth 1 ");
  subdets_.push_back("HB HE HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO Depth 4 ");

  ProblemsVsLB_HB=0;
  ProblemsVsLB_HE=0;
  ProblemsVsLB_HO=0;
  ProblemsVsLB_HF=0;

  NumberOfNeverPresentDigis=0;
  NumberOfNeverPresentDigisHB=0;
  NumberOfNeverPresentDigisHE=0;
  NumberOfNeverPresentDigisHO=0;
  NumberOfNeverPresentDigisHF=0;

  NumberOfNeverPresentRecHits=0;
  NumberOfNeverPresentRecHitsHB=0;
  NumberOfNeverPresentRecHitsHE=0;
  NumberOfNeverPresentRecHitsHO=0;
  NumberOfNeverPresentRecHitsHF=0;

  NumberOfRecentMissingDigis=0;
  NumberOfRecentMissingDigisHB=0;
  NumberOfRecentMissingDigisHE=0;
  NumberOfRecentMissingDigisHO=0;
  NumberOfRecentMissingDigisHF=0;

  NumberOfRecentMissingRecHits=0;
  NumberOfRecentMissingRecHitsHB=0;
  NumberOfRecentMissingRecHitsHE=0;
  NumberOfRecentMissingRecHitsHO=0;
  NumberOfRecentMissingRecHitsHF=0;

  return;
} // void HcalDeadCellClient::init(...)

HcalDeadCellClient::~HcalDeadCellClient()
{
  this->cleanup();
} // destructor

void HcalDeadCellClient::beginJob()
{
  if ( debug_>1 ) std::cout << "HcalDeadCellClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  if (!dbe_) return;
  stringstream mydir;
  mydir<<rootFolder_<<"/DeadCellMonitor_Hcal";
  dbe_->setCurrentFolder(mydir.str().c_str());
  ProblemCells=dbe_->book2D(" ProblemDeadCells",
			   " Problem Dead Cell Rate for all HCAL;i#eta;i#phi",
			   85,-42.5,42.5,
			   72,0.5,72.5);
  SetEtaPhiLabels(ProblemCells);
  mydir<<"/problem_deadcells";
  dbe_->setCurrentFolder(mydir.str().c_str());
  ProblemCellsByDepth.setup(dbe_," Problem Dead Cell Rate");

  return;
} // void HcalDeadCellClient::beginJob()


void HcalDeadCellClient::beginRun(const EventSetup& eventSetup)
{
  if ( debug_>1 ) std::cout << "HcalDeadCellClient: beginRun" << std::endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
} // void HcalDeadCellClient::beginRun(void)


void HcalDeadCellClient::endJob(void)
{
  if ( debug_>1 ) std::cout << "HcalDeadCellClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();
  return;
} // void HcalDeadCellClient::endJob(void)


void HcalDeadCellClient::endRun(std::map<HcalDetId, unsigned int>& myqual) 
{
  if ( debug_>1 ) std::cout << "HcalDeadCellClient: endRun, jevt = " << jevt_ << std::endl;
  calculateProblems(); // calculate problems before root file is closed
  updateChannelStatus(myqual);

  this->cleanup();
  return;
} // void HcalDeadCellClient::endRun(void)

void HcalDeadCellClient::updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual)
{
  if (!dump2database_) return;
  float binval;
  int ieta=0;
  int iphi=0;
  int etabins=0;
  int phibins=0;
  
  int subdet=0;
  stringstream subdetname;
  if (debug_>1)
    {
      std::cout <<"<HcalDeadCellClient>  Summary of Dead Cells in Run: "<<std::endl;
      std::cout <<"(Error rate must be >= "<<minErrorFlag_*100.<<"% )"<<std::endl;  
    }
  for (int d=0;d<4;++d)
    {
      etabins=(ProblemCellsByDepth.depth[d]->getTH2F())->GetNbinsX();
      phibins=(ProblemCellsByDepth.depth[d]->getTH2F())->GetNbinsY();
      for (int hist_eta=0;hist_eta<etabins;++hist_eta)
	{
	  ieta=CalcIeta(hist_eta,d+1);
	  if (ieta==-9999) continue;
	  for (int hist_phi=0;hist_phi<phibins;++hist_phi)
	    {
	      iphi=hist_phi+1;
	      
	      // ProblemCells have already been normalized
	      binval=ProblemCellsByDepth.depth[d]->getBinContent(hist_eta+1,hist_phi+1);
	      
	      // Set subdetector labels for output
	      if (d<2)
		{
		  if (isHB(hist_eta,d+1)) 
		    {
		      subdetname <<"HB";
		      subdet=1;
		    }
		  else if (isHE(hist_eta,d+1)) 
		    {
		      subdetname<<"HE";
		      subdet=2;
		    }
		  else if (isHF(hist_eta,d+1)) 
		    {
		      subdetname<<"HF";
		      subdet=4;
		    }
		}
	      else if (d==2) 
		{
		  subdetname <<"HE";
		  subdet=2;
		}
	      else if (d==3) 
		{
		  subdetname<<"HO";
		  subdet=3;
		}
	      // Set correct depth label
	      
	      HcalDetId myid((HcalSubdetector)(subdet), ieta, iphi, d+1);
	      // Need this to keep from flagging non-existent HE/HF cells
	      if (!validDetId((HcalSubdetector)(subdet), ieta, iphi, d+1))
		continue;

	      int deadcell=0;
	      if (binval>minErrorFlag_)
		deadcell=1;
	      if (deadcell==1 && debug_>0)
		std::cout <<"Dead Cell :  subdetector = "<<subdet<<" (eta,phi,depth) = ("<<ieta<<", "<<iphi<<", "<<d+1<<"):  "<<binval*100.<<"%"<<std::endl;
	      
	      // DetID not found in quality list; add it.  (This shouldn't happen!)
	      if (myqual.find(myid)==myqual.end())
		{
		  myqual[myid]=(deadcell<<HcalChannelStatus::HcalCellDead);  // deadcell shifted to bit 6
		}
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


void HcalDeadCellClient::setup(void) 
{
  return;
} // void HcalDeadCellClient::setup(void)


void HcalDeadCellClient::cleanup(void) 
{
  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  return; // leave deletions to framework
} // void HcalDeadCellClient::cleanup(void)


void HcalDeadCellClient::report()
{
  if(!dbe_) return;
  if ( debug_>1 ) std::cout << "HcalDeadCellClient: report" << std::endl;
  this->setup();

  getHistograms();

  return;
} // HcalDeadCellClient::report()


void HcalDeadCellClient::getHistograms(bool getall)
{
  if(!dbe_) return;

  stringstream name;
  name<<process_.c_str()<<rootFolder_<<"/DeadCellMonitor_Hcal/Dead Cell Task Event Number";
  // Get ievt_ value
  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_>1 ) std::cout << "Found '" << name.str().c_str() << "'" << std::endl;
  }

  // Grab individual histograms
  name.str("");

  getEtaPhiHists(rootFolder_,"DeadCellMonitor_Hcal/dead_digi_never_present/",   "Digi Present At Least Once", DigiPresentByDepth);
  if (deadclient_test_digis_) getEtaPhiHists(rootFolder_,"DeadCellMonitor_Hcal/dead_digi_often_missing/",   "Dead Cells with No Digis", RecentMissingDigisByDepth);
  if (deadclient_test_rechits_)
    {
      getEtaPhiHists(rootFolder_,"DeadCellMonitor_Hcal/dead_rechit_often_missing/",   "RecHits Failing Energy Threshold Test", RecentMissingRecHitsByDepth);
      getEtaPhiHists(rootFolder_,"DeadCellMonitor_Hcal/dead_rechit_neverpresent/",
		     "RecHit Above Threshold At Least Once", RecHitsPresentByDepth);
    }


  // Getting these histograms causes memory leak for some reason.  Hmm...
  if (!getall)
    return;
  // Summary of all dead cells
  ProblemsVsLB=getTProfile("DeadCellMonitor_Hcal/TotalDeadCells_HCAL_vs_LS",
				process_,rootFolder_,dbe_,debug_,cloneME_);
  ProblemsVsLB_HB=getTProfile("DeadCellMonitor_Hcal/TotalDeadCells_HB_vs_LS",
				  process_,rootFolder_,dbe_,debug_,cloneME_);
  ProblemsVsLB_HE=getTProfile("DeadCellMonitor_Hcal/TotalDeadCells_HE_vs_LS",
				  process_,rootFolder_,dbe_,debug_,cloneME_);
  ProblemsVsLB_HO=getTProfile("DeadCellMonitor_Hcal/TotalDeadCells_HO_vs_LS",
				  process_,rootFolder_,dbe_,debug_,cloneME_);
  ProblemsVsLB_HF=getTProfile("DeadCellMonitor_Hcal/TotalDeadCells_HF_vs_LS",
				  process_,rootFolder_,dbe_,debug_,cloneME_);

  // Dead cells -- never present
  NumberOfNeverPresentDigis=getTProfile("DeadCellMonitor_Hcal/dead_digi_never_present/Problem_NeverPresentDigis_HCAL_vs_LS",
					process_,rootFolder_,dbe_,debug_,cloneME_);
  NumberOfNeverPresentDigisHB=getTProfile("DeadCellMonitor_Hcal/dead_digi_never_present/Problem_NeverPresentDigis_HB_vs_LS",
					  process_,rootFolder_,dbe_,debug_,cloneME_);
  NumberOfNeverPresentDigisHE=getTProfile("DeadCellMonitor_Hcal/dead_digi_never_present/Problem_NeverPresentDigis_HE_vs_LS",
					  process_,rootFolder_,dbe_,debug_,cloneME_);
  NumberOfNeverPresentDigisHO=getTProfile("DeadCellMonitor_Hcal/dead_digi_never_present/Problem_NeverPresentDigis_HO_vs_LS",
					  process_,rootFolder_,dbe_,debug_,cloneME_);
  NumberOfNeverPresentDigisHF=getTProfile("DeadCellMonitor_Hcal/dead_digi_never_present/Problem_NeverPresentDigis_HF_vs_LS",
					  process_,rootFolder_,dbe_,debug_,cloneME_);

  // Dead cells -- low occupancy
  if (deadclient_test_digis_)
    {
      NumberOfRecentMissingDigis=getTProfile("DeadCellMonitor_Hcal/dead_digi_often_missing/Problem_RecentMissingDigis_HCAL_vs_LS",
					  process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfRecentMissingDigisHB=getTProfile("DeadCellMonitor_Hcal/dead_digi_often_missing/Problem_RecentMissingDigis_HB_vs_LS",
					    process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfRecentMissingDigisHE=getTProfile("DeadCellMonitor_Hcal/dead_digi_often_missing/Problem_RecentMissingDigis_HE_vs_LS",
					    process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfRecentMissingDigisHO=getTProfile("DeadCellMonitor_Hcal/dead_digi_often_missing/Problem_RecentMissingDigis_HO_vs_LS",
					    process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfRecentMissingDigisHF=getTProfile("DeadCellMonitor_Hcal/dead_digi_often_missing/Problem_RecentMissingDigis_HF_vs_LS",
					    process_,rootFolder_,dbe_,debug_,cloneME_);
    }

  // Dead cells -- low energy
  if (deadclient_test_rechits_)
    {
      NumberOfRecentMissingRecHits=getTProfile("DeadCellMonitor_Hcal/dead_rechit_often_missing/Problem_BelowEnergyRecHits_HCAL_vs_LS",
					   process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfRecentMissingRecHitsHB=getTProfile("DeadCellMonitor_Hcal/dead_rechit_often_missing/Problem_BelowEnergyRecHits_HB_vs_LS",
					     process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfRecentMissingRecHitsHE=getTProfile("DeadCellMonitor_Hcal/dead_rechit_often_missing/Problem_BelowEnergyRecHits_HE_vs_LS",
					     process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfRecentMissingRecHitsHO=getTProfile("DeadCellMonitor_Hcal/dead_rechit_often_missing/Problem_BelowEnergyRecHits_HO_vs_LS",
					     process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfRecentMissingRecHitsHF=getTProfile("DeadCellMonitor_Hcal/dead_rechit_often_missing/Problem_BelowEnergyRecHits_HF_vs_LS",
					     process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfNeverPresentRecHits=getTProfile(
						  "DeadCellMonitor_Hcal/dead_rechit_neverpresent/Problem_RecHitsNeverPresent_HCAL_vs_LS",
						  process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfNeverPresentRecHitsHB=getTProfile(
						    "DeadCellMonitor_Hcal/dead_rechit_neverpresent/Problem_RecHitsNeverPresent_HB_vs_LS",
						    process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfNeverPresentRecHitsHE=getTProfile(
						    "DeadCellMonitor_Hcal/dead_rechit_neverpresent/Problem_RecHitsNeverPresent_HE_vs_LS",
						    process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfNeverPresentRecHitsHO=getTProfile("DeadCellMonitor_Hcal/dead_rechit_neverpresent/Problem_RecHitsNeverPresent_HO_vs_LS",
					     process_,rootFolder_,dbe_,debug_,cloneME_);
      NumberOfNeverPresentRecHitsHF=getTProfile("DeadCellMonitor_Hcal/dead_rechit_neverpresent/Problem_RecHitsNeverPresent_HF_vs_LS",
					     process_,rootFolder_,dbe_,debug_,cloneME_);

    }

  return;
} //void HcalDeadCellClient::getHistograms()


void HcalDeadCellClient::analyze(void)
{
  jevt_++;
  if ( jevt_ % 10 == 0 ) 
    {
      if ( debug_>1 ) std::cout << "<HcalDeadCellClient::analyze>  Running analyze "<<std::endl;
    }
  // Calculate problem cell rate
  calculateProblems();
  return;
} // void HcalDeadCellClient::analyze(void)

void HcalDeadCellClient::calculateProblems()
{
  getHistograms(); 

  double totalevents=0;
  int etabins=0, phibins=0, zside=0;
  double problemvalue=0;

  // Clear away old problems
  if (ProblemCells!=0)
    {
      ProblemCells->Reset();
      (ProblemCells->getTH2F())->SetMaximum(1.);
      (ProblemCells->getTH2F())->SetMinimum(0.);
    }
  for  (unsigned int d=0;d<ProblemCellsByDepth.depth.size();++d)
    if (ProblemCellsByDepth.depth[d]!=0) 
     {
       ProblemCellsByDepth.depth[d]->Reset();
       (ProblemCellsByDepth.depth[d]->getTH2F())->SetMaximum(1.);
       (ProblemCellsByDepth.depth[d]->getTH2F())->SetMinimum(0.);
     }

  // Because we're clearing and re-forming the problem cell histogram here, we don't need to do any cute
  // setting of the underflow bin to 0, and we can plot results as a raw rate between 0-1.
  
  for (unsigned int d=0;d<ProblemCellsByDepth.depth.size();++d)
    {
      if (ProblemCellsByDepth.depth[d]==0) continue;
      if (DigiPresentByDepth[d]==0) continue;
      // Get number of entries from DigiPresent histogram 
      // (need to do this for offline DQM combinations of output)
      totalevents=DigiPresentByDepth[d]->GetBinContent(0);
      if (totalevents==0) continue;
      if (Online_ && totalevents < deadclient_checkNevents_) continue;
      etabins=(ProblemCellsByDepth.depth[d]->getTH2F())->GetNbinsX();
      phibins=(ProblemCellsByDepth.depth[d]->getTH2F())->GetNbinsY();
      problemvalue=0;
      for (int eta=0;eta<etabins;++eta)
	{
	  int ieta=CalcIeta(eta,d+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      problemvalue=0;

	      // Never-present histogram is a boolean, with underflow bin = 1 (for each instance)
	      // Offline DQM adds up never-present histograms from multiple outputs
	      // For now, we want offline DQM to show LS-based 'occupancies', rather than simple boolean on/off
	      // May change in the future?

	      // If cell is never-present in all runs, then problemvalue = event
	      if (DigiPresentByDepth[d]!=0 && DigiPresentByDepth[d]->GetBinContent(eta+1,phi+1)==0) 
		problemvalue=totalevents;
	      // Rec Hit presence test
	      else if (deadclient_test_rechits_ && RecHitsPresentByDepth[d]!=0)
		{
		  if (RecHitsPresentByDepth[d]->GetBinContent(eta+1,phi+1)==0)
		    problemvalue=totalevents;
		  else if (RecHitsPresentByDepth[d]->GetBinContent(eta+1,phi+1)>1)
		    RecHitsPresentByDepth[d]->SetBinContent(eta+1,phi+1,1);
		}
	      else
		{
		  if (deadclient_test_digis_ && RecentMissingDigisByDepth[d]!=0)
		      problemvalue+=RecentMissingDigisByDepth[d]->GetBinContent(eta+1,phi+1);
		  if (deadclient_test_rechits_ && RecentMissingRecHitsByDepth[d]!=0)
		      problemvalue+=RecentMissingRecHitsByDepth[d]->GetBinContent(eta+1,phi+1);
		}
	      if (problemvalue==0) continue;
	      problemvalue/=totalevents; // problem value is a rate; should be between 0 and 1
	      problemvalue = min(1.,problemvalue);
	      
	      zside=0;
	      if (isHF(eta,d+1)) // shift ieta by 1 for HF
		ieta<0 ? zside = -1 : zside = 1;


	      ProblemCellsByDepth.depth[d]->setBinContent(eta+1,phi+1,problemvalue);
	      if (ProblemCells!=0) ProblemCells->Fill(ieta+zside,phi+1,problemvalue);
	    } // loop on phi
	} // loop on eta
    } // loop on depth

  if (ProblemCells==0)
    {
      if (debug_>0) std::cout <<"<HcalDeadCellClient::analyze> ProblemCells histogram does not exist!"<<endl;
      return;
    }

  // We should be able to normalize histograms here, rather than in summary client

  etabins=(ProblemCells->getTH2F())->GetNbinsX();
  phibins=(ProblemCells->getTH2F())->GetNbinsY();
  for (int eta=0;eta<etabins;++eta)
    {
      for (int phi=0;phi<phibins;++phi)
	{
	  if (ProblemCells->getBinContent(eta+1,phi+1)>1.)
	    ProblemCells->setBinContent(eta+1,phi+1,1.);
	}
    }

  return;
} // void HcalDeadCellClient::analyze(void)

void HcalDeadCellClient::createTests()
{
  // Removed a bunch of code that was in older versions of HcalDeadCellClient
  // tests should now be handled from outside
  if(!dbe_) return;
  return;
} // void HcalDeadCellClient::createTests()


void HcalDeadCellClient::resetAllME()
{
  if(!dbe_) return;
  
  stringstream name;

  // Reset individual histograms
  name<<process_.c_str()<<"DeadCellMonitor_Hcal/ ProblemCells";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  for (int i=0;i<4;++i)
    {
      // Reset arrays of histograms
      // Problem Pedestal Plots
      name<<process_.c_str()<<rootFolder_<<"DeadCellMonitor_Hcal/problem_deadcells/"<<subdets_[i]<<" Problem Dead Cell Rate";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"DeadCellMonitor_Hcal/dead_digi_never_present/"<<subdets_[i]<<"Dead Cells with No Digis Ever";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      if (deadclient_test_digis_)
	{
	  name<<process_.c_str()<<rootFolder_<<"DeadCellMonitor_Hcal/dead_digi_often_missing/"<<subdets_[i]<<"Dead Cells with No Digis";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (deadclient_test_rechits_)
	{
	  name<<process_.c_str()<<rootFolder_<<"DeadCellMonitor_Hcal/dead_rechit_often_missing"<<subdets_[i]<<"Dead Cells Failing Energy Threshold Test";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}

      resetME((process_+"DeadCellMonitor_Hcal/TotalDeadCells_HCAL").c_str(),dbe_);
      resetME((process_+"DeadCellMonitor_Hcal/TotalDeadCells_HB").c_str(),dbe_);
      resetME((process_+"DeadCellMonitor_Hcal/TotalDeadCells_HE").c_str(),dbe_);
      resetME((process_+"DeadCellMonitor_Hcal/TotalDeadCells_HO").c_str(),dbe_);
      resetME((process_+"DeadCellMonitor_Hcal/TotalDeadCells_HF").c_str(),dbe_);

      resetME((process_+"DeadCellMonitor_Hcal/dead_digi_never_present/Problem_NeverPresentDigis_HCAL").c_str(),dbe_);
      resetME((process_+"DeadCellMonitor_Hcal/dead_digi_never_present/Problem_NeverPresentDigis_HB").c_str(),dbe_);
      resetME((process_+"DeadCellMonitor_Hcal/dead_digi_never_present/Problem_NeverPresentDigis_HE").c_str(),dbe_);
      resetME((process_+"DeadCellMonitor_Hcal/dead_digi_never_present/Problem_NeverPresentDigis_HO").c_str(),dbe_);
      resetME((process_+"DeadCellMonitor_Hcal/dead_digi_never_present/Problem_NeverPresentDigis_HF").c_str(),dbe_);

      if (deadclient_test_digis_)
	{
	  resetME((process_+"DeadCellMonitor_Hcal/dead_digi_often_missing/Problem_RecentMissingDigis_HCAL").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/dead_digi_often_missing/Problem_RecentMissingDigis_HB").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/dead_digi_often_missing/Problem_RecentMissingDigis_HE").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/dead_digi_often_missing/Problem_RecentMissingDigis_HO").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/dead_digi_often_missing/Problem_RecentMissingDigis_HF").c_str(),dbe_);
	}
      if (deadclient_test_rechits_)
	{
	  resetME((process_+"DeadCellMonitor_Hcal/dead_rechit_often_missing/Problem_BelowEnergyCells_HCAL").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/dead_rechit_often_missing/Problem_BelowEnergyCells_HB").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/dead_rechit_often_missing/Problem_BelowEnergyCells_HE").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/dead_rechit_often_missing/Problem_BelowEnergyCells_HO").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/dead_rechit_often_missing/Problem_BelowEnergyCells_HF").c_str(),dbe_);
	}
    }
  return;
} // void HcalDeadCellClient::resetAllME()


void HcalDeadCellClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  getHistograms(true); 
  if (debug_>1) std::cout << "Preparing HcalDeadCellClient html output ..." << std::endl;

  string client = "DeadCellMonitor";

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal Dead Cell Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Dead Cells</span></h2> " << std::endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<h2><strong>Hcal Dead Cell Status</strong></h2>" << std::endl;
  htmlFile << "<h3>" << std::endl;
  htmlFile << "</h3>" << std::endl;

  htmlFile << "<table align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  htmlAnyHisto(runNo,(ProblemCells->getTH2F()),"i#eta","i#phi", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile<<"<tr align=\"center\"><td> A cell is considered dead if it meets any of the following criteria:"<<std::endl;
  htmlFile<<"<br> A cell's digi is never present during the run;"<<std::endl;
  if (deadclient_test_digis_) htmlFile<<"<br> A cell's digi is not present for "<<deadclient_checkNevents_<<" consecutive events; "<<std::endl;
  if (deadclient_test_rechits_   ) htmlFile<<"<br> A cell's energy is consistently less than a threshold value;"<<std::endl;

  htmlFile<<"</td>"<<std::endl;
  htmlFile<<"</tr></table>"<<std::endl;
  htmlFile<<"<hr><table align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile<<"<tr><td align=center><a href=\"Expert_"<< htmlName<<"\"><h2>Detailed Dead Cell Plots</h2> </a></br></td>"<<std::endl;
  htmlFile<<"</tr></table><br><hr>"<<std::endl;
  
  // Now print out problem cells
  htmlFile <<"<br>"<<std::endl;
  htmlFile << "<h2><strong>Hcal Problem Cells</strong></h2>" << std::endl;
  htmlFile << "(A problem cell is listed below if its failure rate exceeds "<<(100.*minErrorFlag_)<<"%).<br><br>"<<std::endl;
  htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile <<"<td> Problem Dead Cells<br>(ieta, iphi, depth)</td><td align=\"center\"> Fraction of Events <br>in which cells are bad (%)</td></tr>"<<std::endl;

  if (ProblemCells==0)
    {
      if (debug_) std::cout <<"<HcalDeadCellClient::htmlOutput>  ERROR: can't find Problem Dead Cell plot!"<<std::endl;
      // html page footer
      htmlFile <<"</table> " << std::endl;
      htmlFile << "</body> " << std::endl;
      htmlFile << "</html> " << std::endl;

      htmlFile.close();
      return;
    }
  int etabins  = 0;
  int phibins  = 0;
  int ieta=-9999,iphi=-9999;

  stringstream name;
  for (int depth=0;depth<4; ++depth)
    {
      etabins  = (ProblemCells->getTH2F())->GetNbinsX();
      phibins  = (ProblemCells->getTH2F())->GetNbinsY();
      for (int eta=0;eta<etabins;++eta)
        {
	  ieta=CalcIeta(eta, depth+1);
	  if (ieta==-9999) continue;
	  for (int phi=0; phi<phibins;++phi)
            {
              iphi=phi+1;
	      if (abs(ieta)>20 && iphi%2!=1) continue;
	      if (abs(ieta)>39 && iphi%4!=3) continue;
	      if (ProblemCellsByDepth.depth[depth]==0)
		  continue;
	      if (ProblemCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1)>minErrorFlag_)
		{
		  if (depth<2)
		    {
		      if (isHB(eta,depth+1)) name <<"HB";
		      else if (isHE(eta,depth+1)) name<<"HE";
		      else if (isHF(eta,depth+1)) name<<"HF";
		    }
		  else if (depth==2) name <<"HE";
		  else if (depth==3) name<<"HO";

		  htmlFile<<"<td>"<<name.str().c_str()<<" ("<<ieta<<", "<<iphi<<", "<<depth+1<<")</td><td align=\"center\">"<<ProblemCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1)*100.<<"</td></tr>"<<std::endl;

		  name.str("");
		}
	    } // for (int phi=0;...)
	} // for (int eta=0;...)
    } // for (int depth=0;...)
  
  
  // html page footer
  htmlFile <<"</table> " << std::endl;
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;

  htmlFile.close();
  htmlExpertOutput(runNo, htmlDir, htmlName);

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellClient HTMLOUTPUT  -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} //void HcalDeadCellClient::htmlOutput(int runNo, ...) 


void HcalDeadCellClient::htmlExpertOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>1) 
    std::cout <<" <HcalDeadCellClient::htmlExpertOutput>  Preparing Expert html output ..." <<std::endl;
  
  string client = "DeadCellMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_); // does this do anything?

  ofstream htmlFile;
  htmlFile.open((htmlDir +"Expert_"+ htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal Dead Cell Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile <<"<a name=\"EXPERT_DEADCELL_TOP\" href = \".\"> Back to Main HCAL DQM Page </a><br>"<<std::endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Dead Cell Status Page </a><br>"<<std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Dead Cells</span></h2> " << std::endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<table width=100%  border = 1>"<<std::endl;
  htmlFile << "<tr><td align=\"center\" colspan=1><a href=\"#OVERALL_PROBLEMS\">PROBLEM CELLS BY DEPTH </a></td></tr>"<<std::endl;
  htmlFile << "<tr><td align=\"center\">"<<std::endl;
  htmlFile<<"<br><a href=\"#OFF_PROBLEMS\">Dead cell according to Digi Never Present Test </a>"<<std::endl;
  if (deadclient_test_digis_) htmlFile<<"<br><a href=\"#OCC_PROBLEMS\">Dead cell according to Digi Occupancy Test </a>"<<std::endl;
  if (deadclient_test_rechits_   ) htmlFile<<"<br><a href=\"#ENERGY_PROBLEMS\">Dead cell according to Energy Threshold Test </a>"<<std::endl;
  htmlFile << "</td></tr>"<<std::endl;
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><br>"<<std::endl;


  // Plot overall errors
  htmlFile << "<h2><strong><a name=\"OVERALL_PROBLEMS\">Eta-Phi Maps of Problem Cells By Depth</strong></h2>"<<std::endl;
  htmlFile <<" These plots of problem cells combine results from all dead cell tests<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DEADCELL_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,(ProblemCellsByDepth.depth[2*i]->getTH2F()),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,(ProblemCellsByDepth.depth[2*i+1]->getTH2F()),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }

  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;
  
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,ProblemsVsLB,"Number of Dead Cells","Number of occurrences", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,ProblemsVsLB_HB,"Number of Dead Cells","Number of occurrences", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,ProblemsVsLB_HE,"Number of Dead Cells","Number of occurrences", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,ProblemsVsLB_HO,"Number of Dead Cells","Number of occurrences", 92, htmlFile, htmlDir); 
  htmlAnyHisto(runNo,ProblemsVsLB_HF,"Number of Dead Cells","Number of occurrences", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;

  // Dead cells failing digi occupancy tests
  htmlFile << "<h2><strong><a name=\"OFF_PROBLEMS\">Digi Never-Present Problems</strong></h2>"<<std::endl;
  htmlFile <<"A cell fails this test if its digi is never present during the run <br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DEADCELL_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,DigiPresentByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiPresentByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;
  
  // Dead cells failing digi occupancy tests
  if (deadclient_test_digis_)
    {
      htmlFile << "<h2><strong><a name=\"OCC_PROBLEMS\">Digi Occupancy Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its digi is absent for "<<deadclient_checkNevents_<<" consecutive events<br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_DEADCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<2;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,RecentMissingDigisByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,RecentMissingDigisByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	}
      htmlFile <<"</table>"<<std::endl;
      htmlFile <<"<br><hr><br>"<<std::endl;
    }

  // Dead cells failing energy tests
  if (deadclient_test_rechits_)
    {
      htmlFile << "<h2><strong><a name=\"ENERGY_PROBLEMS\">Energy Threshold Test Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its rechit energy is below threshold for "<<deadclient_checkNevents_<<" consecutive events <br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_DEADCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<2;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,RecentMissingRecHitsByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,RecentMissingRecHitsByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	}

      htmlFile <<"</table>"<<std::endl;
      htmlFile <<"<br><hr><br>"<<std::endl;
    }

  htmlFile <<"<br><hr><br><a href= \"#EXPERT_DEADCELL_TOP\" > Back to Top of Page </a><br>"<<std::endl;
  htmlFile <<"<a href = \".\"> Back to Main HCAL DQM Page </a><br>"<<std::endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Dead Cell Status Page </a><br>"<<std::endl;

  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  
  htmlFile.close();

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellClient  HTMLEXPERTOUTPUT ->"<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalDeadCellClient::htmlExpertOutput(...)



void HcalDeadCellClient::loadHistograms(TFile* infile)
{
  // deprecated function; no longer needed
  return;
} // void HcalDeadCellClient::loadHistograms(...)




bool HcalDeadCellClient::hasErrors_Temp()
{
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
	      if (ProblemCellsByDepth.depth[depth]==0)
		{
		  continue;
		}
	      if (ProblemCellsByDepth.depth[depth]->getBinContent(hist_eta,hist_phi)>minErrorFlag_)
		{
		  problemcount++;
		}
	    } // for (int hist_phi=1;...)
	} // for (int hist_eta=1;...)
    } // for (int depth=0;...)

  if (problemcount>=100) return true;
  return false;
} // bool HcalDeadCellClient::hasErrors_Temp()

bool HcalDeadCellClient::hasWarnings_Temp()
{
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
	      if (ProblemCellsByDepth.depth[depth]==0)
		{
		  continue;
		}
	      if (ProblemCellsByDepth.depth[depth]->getBinContent(hist_eta,hist_phi)>minErrorFlag_)
		{
		  problemcount++;
		}
	    } // for (int hist_phi=1;...)
	} // for (int hist_eta=1;...)
    } // for (int depth=0;...)

  if (problemcount>0) return true;
  return false;

} // bool HcalDeadCellClient::hasWarnings_Temp()
