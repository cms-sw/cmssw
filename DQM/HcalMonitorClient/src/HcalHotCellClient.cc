#include <DQM/HcalMonitorClient/interface/HcalHotCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <math.h>
#include <iostream>

#define BITSHIFT 6

HcalHotCellClient::HcalHotCellClient(){} // constructor 

HcalHotCellClient::~HcalHotCellClient()
{
  //this->cleanup();
} // destructor

void HcalHotCellClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName)
{
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  // Get variable values from cfg file
  // Set which hot cell checks will looked at
  hotclient_test_persistent_         = ps.getUntrackedParameter<bool>("HotCellClient_test_persistent",false);
  hotclient_test_pedestal_          = ps.getUntrackedParameter<bool>("HotCellClient_test_pedestal",false);
  hotclient_test_neighbor_          = ps.getUntrackedParameter<bool>("HotCellClient_test_neighbor",false);
  hotclient_test_energy_            = ps.getUntrackedParameter<bool>("HotCellClient_test_energy",false);

  hotclient_checkNevents_ = ps.getUntrackedParameter<int>("HotCellClient_checkNevents",100);

  minErrorFlag_ = ps.getUntrackedParameter<double>("HotCellClient_minErrorFlag",0.0);

  hotclient_makeDiagnostics_ = ps.getUntrackedParameter<bool>("HotCellClient_makeDiagnosticPlots",false);

  dump2database_ = false; // eventually, make this a configurable boolean
  
  // Set histograms to NULL
  ProblemCells=0;
  for (int i=0;i<4;++i)
    {
      // Set each array's pointers to NULL
      AbovePersistentThresholdCellsByDepth[i] =0;
      AbovePedestalHotCellsByDepth[i]         =0;
      AboveNeighborsHotCellsByDepth[i]        =0;
      AboveEnergyThresholdCellsByDepth[i]     =0;
      d_avgrechitenergymap[i]                 =0;
      d_avgrechitoccupancymap[i]              =0;
    }  

  if (hotclient_makeDiagnostics_)
    {
      d_HBnormped=0;
      d_HBrechitenergy=0;
      d_HBenergyVsNeighbor=0;
      d_HEnormped=0;
      d_HErechitenergy=0;
      d_HEenergyVsNeighbor=0;
      d_HOnormped=0;
      d_HOrechitenergy=0;
      d_HOenergyVsNeighbor=0;
      d_HFnormped=0;
      d_HFrechitenergy=0;
      d_HFenergyVsNeighbor=0;
    } // if (hotclient_makeDiagnostics_)

  subdets_.push_back("HBE HE HF Depth 1 ");
  subdets_.push_back("HB HE HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO Depth 4 ");


  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient INIT -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalHotCellClient::init(...)

void HcalHotCellClient::beginJob(const EventSetup& eventSetup)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if ( debug_>1 ) std::cout << "HcalHotCellClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;
  this->setup();

  stringstream mydir;
  mydir<<rootFolder_<<"/HotCellMonitor_Hcal";
  dbe_->setCurrentFolder(mydir.str().c_str());
  ProblemCells=dbe_->book2D(" ProblemHotCells",
			   " Problem Hot Cell Rate for all HCAL;i#eta;i#phi",
			   85,-42.5,42.5,
			   72,0.5,72.5);
  SetEtaPhiLabels(ProblemCells);
  mydir<<"/problem_hotcells";
  dbe_->setCurrentFolder(mydir.str().c_str());
  ProblemCellsByDepth.setup(dbe_," Problem Hot Cell Rate");

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient BEGINJOB -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalHotCellClient::beginJob(const EventSetup& eventSetup);


void HcalHotCellClient::beginRun(void)
{
  if ( debug_>1 ) std::cout << "HcalHotCellClient: beginRun" << std::endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient BEGINRUN -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalHotCellClient::beginRun(void)


void HcalHotCellClient::endJob(std::map<HcalDetId, unsigned int>& myqual) 
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if ( debug_>1 ) std::cout << "HcalHotCellClient: endJob, ievt = " << ievt_ << std::endl;

  // Write to database at end of run, or end of job?
  if (dump2database_==true) // don't do anything special unless specifically asked to dump db file
    {
      float binval;
      int ieta=0;
      int iphi=0;
      int etabins=0;
      int phibins=0;

      int subdet=0;
      ostringstream subdetname;
      if (debug_>1)
	{
	  std::cout <<"<HcalHotCellClient>  Summary of Hot Cells in Run: "<<std::endl;
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
		  if (binval<=minErrorFlag_)
		    continue;
		  if (debug_>0)
		    std::cout <<"Hot Cell "<<subdet<<"("<<ieta<<", "<<iphi<<", "<<d+1<<"):  "<<binval*100.<<"%"<<std::endl;

		  // if we've reached here, hot cell condition was met
		  int value=1;

		  if (myqual.find(myid)==myqual.end())
		    {
		      myqual[myid]=(value<<BITSHIFT);  // hotcell shifted to bit 6
		    }
		  else
		    {
		      int mask=(1<<BITSHIFT);
		      if (value==1)
			myqual[myid] |=mask;
		  
		      else
			myqual[myid] &=~mask;
		    }
		
		} // for (int hist_phi=1;hist_phi<=phibins;++hist_phi)
	    } // for (int hist_eta=1;hist_eta<=etabins;++hist_eta)
	} // for (int d=0;d<4;++d)
    } // if (dump2database_==true)
  //this->cleanup();
  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient ENDJOB -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalHotCellClient::endJob(void)


void HcalHotCellClient::endRun(void) 
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  calculateProblems();
  // write to DB here as well?
  //this->cleanup();
  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient ENDRUN -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  //calculateProblems();
  return;
} // void HcalHotCellClient::endRun(void)


void HcalHotCellClient::setup(void) 
{
  return;
} // void HcalHotCellClient::setup(void)


void HcalHotCellClient::cleanup(void) 
{
  // seems to cause crashes; remove?
  return;
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient CLEANUP -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalHotCellClient::cleanup(void)


void HcalHotCellClient::report()
{
  if(!dbe_) return;
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if ( debug_>1 ) std::cout << "HcalHotCellClient: report" << std::endl;
  this->setup();

  getHistograms();
  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient REPORT -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // HcalHotCellClient::report()


void HcalHotCellClient::getHistograms()
{
  if(!dbe_) return;
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  ostringstream name;
  name<<process_.c_str()<<rootFolder_<<"/HotCellMonitor_Hcal/Hot Cell Task Event Number";
  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_>1 ) std::cout << "Found '" << name.str().c_str() << "'" << std::endl;
  }
  name.str("");

  // Grab individual histograms

  if (hotclient_test_persistent_) getEtaPhiHists(rootFolder_,"HotCellMonitor_Hcal/hot_rechit_always_above_threshold/",   "Hot Cells Persistently Above Energy Threshold", AbovePersistentThresholdCellsByDepth);
  if (hotclient_test_pedestal_)  getEtaPhiHists(rootFolder_,"HotCellMonitor_Hcal/hot_pedestaltest/", "Hot Cells Above Pedestal", AbovePedestalHotCellsByDepth);
  if (hotclient_test_neighbor_)  getEtaPhiHists(rootFolder_,"HotCellMonitor_Hcal/hot_neighbortest/", "Hot Cells Failing Neighbor Test", AboveNeighborsHotCellsByDepth);
  if (hotclient_test_energy_)    getEtaPhiHists(rootFolder_,"HotCellMonitor_Hcal/hot_rechit_above_threshold/",   "Hot Cells Above Energy Threshold", AboveEnergyThresholdCellsByDepth);

  if (hotclient_makeDiagnostics_)
    {
      getEtaPhiHists(rootFolder_,"HotCellMonitor_Hcal/diagnostics/rechitenergy/","Rec hit energy per cell",d_avgrechitenergymap);
      getEtaPhiHists(rootFolder_,"HotCellMonitor_Hcal/diagnostics/rechitenergy/","Rec hit occupancy per cell",d_avgrechitoccupancymap);

      // At some point, clean these up so that histograms are only retrieved if corresponding process ran in Task
      d_HBnormped=getTH1F("HotCellMonitor_Hcal/diagnostics/pedestal/HB_normped", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HBrechitenergy=getTH1F("HotCellMonitor_Hcal/diagnostics/rechitenergy/HB_rechitenergy", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HBenergyVsNeighbor=getTH2F("HotCellMonitor_Hcal/diagnostics/neighborcells/HB_energyVsNeighbor", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HEnormped=getTH1F("HotCellMonitor_Hcal/diagnostics/pedestal/HE_normped", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HErechitenergy=getTH1F("HotCellMonitor_Hcal/diagnostics/rechitenergy/HE_rechitenergy", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HEenergyVsNeighbor=getTH2F("HotCellMonitor_Hcal/diagnostics/neighborcells/HE_energyVsNeighbor", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HOnormped=getTH1F("HotCellMonitor_Hcal/diagnostics/pedestal/HO_normped", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HOrechitenergy=getTH1F("HotCellMonitor_Hcal/diagnostics/rechitenergy/HO_rechitenergy", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HOenergyVsNeighbor=getTH2F("HotCellMonitor_Hcal/diagnostics/neighborcells/HO_energyVsNeighbor", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HFnormped=getTH1F("HotCellMonitor_Hcal/diagnostics/pedestal/HF_normped", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HFrechitenergy=getTH1F("HotCellMonitor_Hcal/diagnostics/rechitenergy/HF_rechitenergy", process_, rootFolder_, dbe_, debug_, cloneME_);
      d_HFenergyVsNeighbor=getTH2F("HotCellMonitor_Hcal/diagnostics/neighborcells/HF_energyVsNeighbor", process_, rootFolder_, dbe_, debug_, cloneME_);
    } // if (hotclient_makeDiagnostics_)


  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient GETHISTOGRAMS -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} //void HcalHotCellClient::getHistograms()


void HcalHotCellClient::analyze(void)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  jevt_++;
  if ( jevt_ % 10 == 0 ) 
    {
      if ( debug_>1 ) std::cout << "<HcalHotCellClient::analyze>  Running analyze "<<std::endl;
    }
  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient ANALYZE -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  calculateProblems();
  return;
} // void HcalHotCellClient::analyze(void)

void HcalHotCellClient::calculateProblems()
{
  getHistograms();
  double totalevents=0; // total events processed thus far
  int etabins=0, phibins=0, zside=0;
  double problemvalue=0;

  // Clear away old problems
  if (ProblemCells!=0)
    ProblemCells->Reset();
  for  (unsigned int d=0;d<ProblemCellsByDepth.depth.size();++d)
    if (ProblemCellsByDepth.depth[d]!=0) 
      ProblemCellsByDepth.depth[d]->Reset();

  // Because we're clearing and re-forming the problem cell histogram here, we don't need to do any cute
  // setting of the underflow bin to 0, and we can plot results as a raw rate between 0-1.
  
  for (unsigned int d=0;d<ProblemCellsByDepth.depth.size();++d)
    {
      if (ProblemCellsByDepth.depth[d]==0) continue;
      // All tests have the same number of 'totalevents' stored in their
      // underflow bins.  As long as one test is being performed, grab value
      // from that test.  Otherwise, continue
      if (hotclient_test_persistent_ && AbovePersistentThresholdCellsByDepth[d]!=0)
	totalevents=AbovePersistentThresholdCellsByDepth[d]->GetBinContent(0);
      else if (hotclient_test_pedestal_ && AbovePedestalHotCellsByDepth[d]!=0)
	totalevents=AbovePedestalHotCellsByDepth[d]->GetBinContent(0);
      else if (hotclient_test_neighbor_ && AboveNeighborsHotCellsByDepth[d]!=0)
	totalevents=AboveNeighborsHotCellsByDepth[d]->GetBinContent(0);
      else if (hotclient_test_energy_ && AboveEnergyThresholdCellsByDepth[d]!=0)
	totalevents=AboveEnergyThresholdCellsByDepth[d]->GetBinContent(0);
      else
	continue;

      if (totalevents==0) continue;
      // get number of bins for problemcells
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
	      // Total # of problems = sum of problems from each test
	      if (hotclient_test_persistent_)
		problemvalue+=AbovePersistentThresholdCellsByDepth[d]->GetBinContent(eta+1,phi+1);
	      if (hotclient_test_neighbor_)
		  problemvalue+=AboveNeighborsHotCellsByDepth[d]->GetBinContent(eta+1,phi+1);
	      if (hotclient_test_energy_)
		  problemvalue+=AboveEnergyThresholdCellsByDepth[d]->GetBinContent(eta+1,phi+1);
	      problemvalue = min(totalevents, problemvalue);
	      if (problemvalue==0) continue; // no problem found
	      zside=0;
	      if (d<2)
		{
		  if (isHF(eta,d+1))
		    ieta<0 ? zside = -1 : zside= 1;
		}
	      ProblemCellsByDepth.depth[d]->Fill(ieta+zside,phi+1,problemvalue/totalevents);
	      ProblemCells->Fill(ieta+zside,phi+1,problemvalue/totalevents);
	    } // loop over phi
	} // loop over eta
    } //loop over d

  if (ProblemCells==0)
    {
      if (debug_>0) std::cout <<"<HcalHotCellClient::analyze> ProblemCells histogram does not exist!"<<endl;
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

} // calculateProblems()


void HcalHotCellClient::createTests()
{
  // Removed a bunch of code that was in older versions of HcalHotCellClient
  // tests should now be handled from outside
  if(!dbe_) return;
  return;
} // void HcalHotCellClient::createTests()


void HcalHotCellClient::resetAllME()
{
  if(!dbe_) return;
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  ostringstream name;


  for (int i=0;i<4;++i)
    {
      // Reset arrays of histograms

      if (hotclient_test_persistent_)
	{
	  name<<process_.c_str()<<rootFolder_<<"HotCellMonitor_Hcal/hot_unoccupied_digi/"<<subdets_[i]<<"Hot Cells with No Digis";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (hotclient_test_pedestal_)
	{
	  name<<process_.c_str()<<rootFolder_<<"HotCellMonitor_Hcal/hot_pedestaltest"<<subdets_[i]<<"Hot Cells Failing Pedestal Test";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (hotclient_test_neighbor_)
	{
	  name<<process_.c_str()<<rootFolder_<<"HotCellMonitor_Hcal/hot_neighbortest"<<subdets_[i]<<"Hot Cells Failing Neighbor Test";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (hotclient_test_energy_)
	{
	  name<<process_.c_str()<<rootFolder_<<"HotCellMonitor_Hcal/hot_energytest"<<subdets_[i]<<"Hot Cells Failing Energy Threshold Test";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (hotclient_makeDiagnostics_)
	{
	  resetME("HotCellMonitor_Hcal/diagnostics/pedestal/HB_normped",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/rechitenergy/HB_rechitenergy",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/neighborcells/HB_energyVsNeighbor",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/pedestal/HE_normped",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/rechitenergy/HE_rechitenergy",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/neighborcells/HE_energyVsNeighbor",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/pedestal/HO_normped",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/rechitenergy/HO_rechitenergy",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/neighborcells/HO_energyVsNeighbor",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/pedestal/HF_normped",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/rechitenergy/HF_rechitenergy",dbe_);
	  resetME("HotCellMonitor_Hcal/diagnostics/neighborcells/HF_energyVsNeighbor",dbe_);
	} // if (hotclient_makeDiagnostics_)
      
    } // for (int i=0;i<4;++i)
  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient RESETALLME -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalHotCellClient::resetAllME()


void HcalHotCellClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>1) std::cout << "Preparing HcalHotCellClient html output ..." << std::endl;
  getHistograms();
  string client = "HotCellMonitor";

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal Hot Cell Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Hot Cells</span></h2> " << std::endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<h2><strong>Hcal Hot Cell Status</strong></h2>" << std::endl;
  htmlFile << "<h3>" << std::endl;
  htmlFile << "</h3>" << std::endl;

  htmlFile << "<table align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  htmlAnyHisto(runNo,ProblemCells->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile<<"<tr align=\"center\"><td> A cell is considered hot if it meets any of the following criteria:"<<std::endl;
  if (hotclient_test_persistent_) htmlFile<<"<br> A cell's ADC sum is more than (pedestal + N sigma); "<<std::endl;
  if (hotclient_test_pedestal_ ) htmlFile<<"<br> A cell's energy is above some threshold value X;"<<std::endl;
  if (hotclient_test_energy_   ) htmlFile<<"<br> A cell's energy is consistently above some threshold value Y (where Y does not necessarily equal X);"<<std::endl;
  if (hotclient_test_neighbor_ ) htmlFile<<"<br> A cell's energy is much more than the sum of its neighbors;"<<std::endl;
  htmlFile<<"</td>"<<std::endl;
  htmlFile<<"</tr></table>"<<std::endl;
  htmlFile<<"<hr><table align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile<<"<tr><td align=center><a href=\"Expert_"<< htmlName<<"\"><h2>Detailed Hot Cell Plots</h2> </a></br></td>"<<std::endl;
  htmlFile<<"</tr></table><br><hr>"<<std::endl;
  
  // Now print out problem cells
  htmlFile <<"<br>"<<std::endl;
  htmlFile << "<h2><strong>Hcal Problem Cells</strong></h2>" << std::endl;
  htmlFile << "(A problem cell is listed below if its failure rate exceeds "<<(100.*minErrorFlag_)<<"%).<br><br>"<<std::endl;
  htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile <<"<td> Problem Hot Cells<br>(ieta, iphi, depth)</td><td align=\"center\"> Fraction of Events <br>in which cells are bad (%)</td></tr>"<<std::endl;

  if (ProblemCells==0)
    {
      if (debug_) std::cout <<"<HcalHotCellClient::htmlOutput>  ERROR: can't find Problem Hot Cell plot!"<<std::endl;
      return;
    }
  int ieta=-9999,iphi=-9999;
  int etabins=0, phibins=0;
  ostringstream name;
  for (int depth=0;depth<4; ++depth)
    {
      etabins  = (ProblemCellsByDepth.depth[depth]->getTH2F())->GetNbinsX();
      phibins  = (ProblemCellsByDepth.depth[depth]->getTH2F())->GetNbinsY();
      for (int hist_eta=0;hist_eta<etabins;++hist_eta)
        {
	  ieta=CalcIeta(hist_eta,depth+1);
	  if (ieta==-9999) continue;
          for (int hist_phi=0; hist_phi<phibins;++hist_phi)
            {
              iphi=hist_phi+1;
	      if (abs(ieta)>20 && iphi%2!=1) continue;
	      if (abs(ieta)>39 && iphi%4!=3) continue;
	      
	      if (ProblemCellsByDepth.depth[depth]==0)
		continue;

	      if (ProblemCellsByDepth.depth[depth]->getBinContent(hist_eta+1,hist_phi+1)>minErrorFlag_)
		{
		  if (depth<2)
		    {
		      if (isHB(hist_eta,depth+1)) name <<"HB";
		      else if (isHE(hist_eta,depth+1)) name<<"HE";
		      else if (isHF(hist_eta,depth+1)) name<<"HF";
		    }
		  else if (depth==2) name <<"HE";
		  else if (depth==3) name<<"HO";
		  htmlFile<<"<td>"<<name.str().c_str()<<" ("<<ieta<<", "<<iphi<<", "<<depth+1<<")</td><td align=\"center\">"<<ProblemCellsByDepth.depth[depth]->getBinContent(hist_eta+1,hist_phi+1)*100.<<"</td></tr>"<<std::endl;

		  name.str("");
		}
	    } // for (int hist_phi=0;...)
	} // for (int hist_eta=0;...)
    } // for (int depth=0;...)
  
  
  // html page footer
  htmlFile <<"</table> " << std::endl;
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;

  htmlFile.close();
  htmlExpertOutput(runNo, htmlDir, htmlName);

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient HTMLOUTPUT  -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} //void HcalHotCellClient::htmlOutput(int runNo, ...) 


void HcalHotCellClient::htmlExpertOutput(int runNo, string htmlDir, string htmlName)
{

  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>1) 
    std::cout <<" <HcalHotCellClient::htmlExpertOutput>  Preparing Expert html output ..." <<std::endl;
  
  string client = "HotCellMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_); // does this do anything?

ofstream htmlFile;
  htmlFile.open((htmlDir +"Expert_"+ htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal Hot Cell Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile <<"<a name=\"EXPERT_HOTCELL_TOP\" href = \".\"> Back to Main HCAL DQM Page </a><br>"<<std::endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Hot Cell Status Page </a><br>"<<std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Hot Cells</span></h2> " << std::endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<table width=100%  border = 1>"<<std::endl;
  htmlFile << "<tr><td align=\"center\" colspan=1><a href=\"#OVERALL_PROBLEMS\">PROBLEM CELLS BY DEPTH </a></td></tr>"<<std::endl;
  htmlFile << "<tr><td align=\"center\">"<<std::endl;
  if (hotclient_test_pedestal_ ) htmlFile<<"<br><a href=\"#PED_PROBLEMS\">Hot cell according to Pedestal Test </a>"<<std::endl;
  if (hotclient_test_energy_   ) htmlFile<<"<br><a href=\"#ENERGY_PROBLEMS\">Hot cell according to Energy Threshold Test </a>"<<std::endl;
  if (hotclient_test_persistent_) htmlFile<<"<br><a href=\"#PERSISTENT_PROBLEMS\">Hot cell consistently above a certain energy </a>"<<std::endl;
  if (hotclient_test_neighbor_ ) htmlFile<<"<br><a href=\"#NEIGHBOR_PROBLEMS\">Hot cell according to Neighbor Test </a>"<<std::endl;
  htmlFile << "</td></tr>"<<std::endl;
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><br>"<<std::endl;


  // Plot overall errors
  htmlFile << "<h2><strong><a name=\"OVERALL_PROBLEMS\">Eta-Phi Maps of Problem Cells By Depth</strong></h2>"<<std::endl;
  htmlFile <<" These plots of problem cells combine results from all hot cell tests<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  
  // remap so that HE depths are plotted consecutively
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,(ProblemCellsByDepth.depth[2*i]->getTH2F()),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,(ProblemCellsByDepth.depth[2*i+1]->getTH2F()),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }

  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;
  
 
  // Hot cells failing pedestal tests
  if (hotclient_test_pedestal_)
    {
      htmlFile << "<h2><strong><a name=\"PED_PROBLEMS\">Pedestal Test Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its ADC sum is above (pedestal + Nsigma) for  "<<hotclient_checkNevents_<<" consecutive events <br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<2;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,AbovePedestalHotCellsByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,AbovePedestalHotCellsByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	}
      if (hotclient_makeDiagnostics_)
	{
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HBnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlAnyHisto(runNo, d_HEnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlFile <<"</tr>"<<std::endl;
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HOnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlAnyHisto(runNo, d_HFnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlFile <<"</tr>"<<std::endl;
	} // if (hotclient_makeDiagnostics_)
      htmlFile <<"</table>"<<std::endl;
      htmlFile <<"<br><hr><br>"<<std::endl;
    }

  // Hot cells failing energy tests
  if (hotclient_test_energy_)
    {
      htmlFile << "<h2><strong><a name=\"ENERGY_PROBLEMS\">Energy Threshold Test Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its rechit energy is above threshold at any time.<br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<2;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,AboveEnergyThresholdCellsByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,AboveEnergyThresholdCellsByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	}
      if (hotclient_makeDiagnostics_)
	{
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HBrechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlAnyHisto(runNo, d_HErechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlFile <<"</tr>"<<std::endl;
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HOrechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlAnyHisto(runNo, d_HFrechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlFile <<"</tr>"<<std::endl;
	} // if (hotclient_makeDiagnostics_)

      htmlFile <<"</table>"<<std::endl;
      htmlFile <<"<br><hr><br>"<<std::endl;
    }

  // Hot cells persistently above some threshold energy
  if (hotclient_test_persistent_)
    {
      htmlFile << "<h2><strong><a name=\"PERSISTENT_PROBLEMS\">Persistent Hot Cell Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its rechit energy is above threshold for "<<hotclient_checkNevents_<<" consecutive events.<br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<2;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,AbovePersistentThresholdCellsByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir,0,0);
	  htmlAnyHisto(runNo,AbovePersistentThresholdCellsByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir,0,0);
	  htmlFile <<"</tr>"<<std::endl;
	}
      htmlFile <<"</table>"<<std::endl;
      htmlFile <<"<br><hr><br>"<<std::endl;
    }


  // Hot cells failing neighbor tests
  if (hotclient_test_neighbor_)
    {
      htmlFile << "<h2><strong><a name=\"NEIGHBOR_PROBLEMS\">Neighbor Energy Test Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its rechit energy is significantly greater than the sum of its surrounding neighbors <br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<2;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,AboveNeighborsHotCellsByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,AboveNeighborsHotCellsByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	}
      if (hotclient_makeDiagnostics_)
	{
	  gStyle->SetPalette(1);  // back to rainbow coloring
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HBenergyVsNeighbor, "Cell energy (GeV)","Neighbor energy (GeV)", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo, d_HEenergyVsNeighbor, "Cell energy (GeV)","Neighbor energy (GeV)", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HOenergyVsNeighbor, "Cell energy (GeV)","Neighbor energy (GeV)", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo, d_HFenergyVsNeighbor, "Cell energy (GeV)","Neighbor energy (GeV)", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	} // if (hotclient_makeDiagnostics_)

      htmlFile <<"</table>"<<std::endl;
      htmlFile <<"<br><hr><br>"<<std::endl;
    }


  htmlFile <<"<br><hr><br><a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top of Page </a><br>"<<std::endl;
  htmlFile <<"<a href = \".\"> Back to Main HCAL DQM Page </a><br>"<<std::endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Hot Cell Status Page </a><br>"<<std::endl;

  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  
  htmlFile.close();

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellClient  HTMLEXPERTOUTPUT ->"<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalHotCellClient::htmlExpertOutput(...)

void HcalHotCellClient::loadHistograms(TFile* infile)
{
  // deprecated function; no longer needed
  return;
} // void HcalHotCellClient::loadHistograms(...)

bool HcalHotCellClient::hasErrors_Temp()
{
  int problemcount=0;
  int ieta;

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

} // bool HcalHotCellClient::hasErrors_Temp()

bool HcalHotCellClient::hasWarnings_Temp()
{
  int problemcount=0;
  int ieta=0;
 
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

} // bool HcalHotCellClient::hasWarnings_Temp()
