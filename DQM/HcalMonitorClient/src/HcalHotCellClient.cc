#include <DQM/HcalMonitorClient/interface/HcalHotCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <math.h>
#include <iostream>

HcalHotCellClient::HcalHotCellClient(){} // constructor 

void HcalHotCellClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  // Get variable values from cfg file
  // Set which hot cell checks will looked at
  hotclient_test_persistent_         = ps.getUntrackedParameter<bool>("HotCellClient_test_persistent",true);
  hotclient_test_pedestal_          = ps.getUntrackedParameter<bool>("HotCellClient_test_pedestal",true);
  hotclient_test_neighbor_          = ps.getUntrackedParameter<bool>("HotCellClient_test_neighbor",true);
  hotclient_test_energy_            = ps.getUntrackedParameter<bool>("HotCellClient_test_energy",true);

  hotclient_checkNevents_ = ps.getUntrackedParameter<int>("HotCellClient_checkNevents",100);
  hotclient_checkNevents_persistent_ = ps.getUntrackedParameter<int>("HotCellClient_checkNevents_persistent",hotclient_checkNevents_);
  hotclient_checkNevents_pedestal_  = ps.getUntrackedParameter<int>("HotCellClient_checkNevents_pedestal" ,hotclient_checkNevents_);
  hotclient_checkNevents_neighbor_  = ps.getUntrackedParameter<int>("HotCellClient_checkNevents_neighbor" ,hotclient_checkNevents_);
  hotclient_checkNevents_energy_    = ps.getUntrackedParameter<int>("HotCellClient_checkNevents_energy"   ,hotclient_checkNevents_);

  minErrorFlag_ = ps.getUntrackedParameter<double>("HotCellClient_minErrorFlag",0.0);

  hotclient_makeDiagnostics_ = ps.getUntrackedParameter<bool>("HotCellClient_makeDiagnosticPlots",false);

  // Set histograms to NULL
  ProblemHotCells=0;
  for (int i=0;i<6;++i)
    {
      // Set each array's pointers to NULL
      ProblemHotCellsByDepth[i]               =0;
      AbovePersistentThresholdCellsByDepth[i] =0;
      AbovePedestalHotCellsByDepth[i]         =0;
      AboveNeighborsHotCellsByDepth[i]        =0;
      AboveEnergyThresholdCellsByDepth[i]     =0;
      d_avgrechitenergymap[i]                 =0;
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
      d_ZDCnormped=0;
      d_ZDCrechitenergy=0;
      d_ZDCenergyVsNeighbor=0;
    } // if (hotclient_makeDiagnostics_)

  subdets_.push_back("HB HF Depth 1 ");
  subdets_.push_back("HB HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO ZDC ");
  subdets_.push_back("HE Depth 1 ");
  subdets_.push_back("HE Depth 2 ");

  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient INIT -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalHotCellClient::init(...)


HcalHotCellClient::~HcalHotCellClient()
{
  this->cleanup();
} // destructor


void HcalHotCellClient::beginJob(const EventSetup& eventSetup)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if ( debug_>1 ) cout << "HcalHotCellClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient BEGINJOB -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalHotCellClient::beginJob(const EventSetup& eventSetup);


void HcalHotCellClient::beginRun(void)
{
  if ( debug_>1 ) cout << "HcalHotCellClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient BEGINRUN -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalHotCellClient::beginRun(void)


void HcalHotCellClient::endJob(void) 
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if ( debug_>1 ) cout << "HcalHotCellClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient ENDJOB -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalHotCellClient::endJob(void)


void HcalHotCellClient::endRun(void) 
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if ( debug_>1 ) cout << "HcalHotCellClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient ENDRUN -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalHotCellClient::endRun(void)


void HcalHotCellClient::setup(void) 
{
  return;
} // void HcalHotCellClient::setup(void)


void HcalHotCellClient::cleanup(void) 
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if(cloneME_)
    {
      // delete individual histogram pointers
      if (ProblemHotCells) delete ProblemHotCells;
      
      for (int i=0;i<6;++i)
	{
	  // delete pointers within arrays of histograms
	  if (ProblemHotCellsByDepth[i])               delete ProblemHotCellsByDepth[i];

	  if (AbovePersistentThresholdCellsByDepth[i]) delete AbovePersistentThresholdCellsByDepth[i];
	  if (AbovePedestalHotCellsByDepth[i])         delete AbovePedestalHotCellsByDepth[i];
	  if (AboveNeighborsHotCellsByDepth[i])        delete AboveNeighborsHotCellsByDepth[i];
	  if (AboveEnergyThresholdCellsByDepth[i])     delete AboveEnergyThresholdCellsByDepth[i];
	  if (d_avgrechitenergymap[i])                 delete d_avgrechitenergymap[i];
	}
      
      if (hotclient_makeDiagnostics_)
	{
	  if (d_HBnormped)          delete d_HBnormped;
	  if (d_HBrechitenergy)     delete d_HBrechitenergy;
	  if (d_HBenergyVsNeighbor) delete d_HBenergyVsNeighbor;
	  if (d_HEnormped)          delete d_HEnormped;
	  if (d_HErechitenergy)     delete d_HErechitenergy;
	  if (d_HEenergyVsNeighbor) delete d_HEenergyVsNeighbor;
	  if (d_HOnormped)          delete d_HOnormped;
	  if (d_HOrechitenergy)     delete d_HOrechitenergy;
	  if (d_HOenergyVsNeighbor) delete d_HOenergyVsNeighbor;
	  if (d_HFnormped)          delete d_HFnormped;
	  if (d_HFrechitenergy)     delete d_HFrechitenergy;
	  if (d_HFenergyVsNeighbor) delete d_HFenergyVsNeighbor;
	  if (d_ZDCnormped)         delete d_ZDCnormped;
	  if (d_ZDCrechitenergy)    delete d_ZDCrechitenergy;
	  if (d_ZDCenergyVsNeighbor)delete d_ZDCenergyVsNeighbor;

	} // if (hotclient_makeDiagnostics_)
      

    }

  // Set individual pointers to NULL
  ProblemHotCells = 0;

  for (int i=0;i<6;++i)
    {
      // Set each array's pointers to NULL
      ProblemHotCellsByDepth[i]               =0;
      AbovePersistentThresholdCellsByDepth[i] =0;
      AbovePedestalHotCellsByDepth[i]         =0;
      AboveNeighborsHotCellsByDepth[i]        =0;
      AboveEnergyThresholdCellsByDepth[i]     =0;
      d_avgrechitenergymap[i]                 =0;
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
      d_ZDCnormped=0;
      d_ZDCrechitenergy=0;
      d_ZDCenergyVsNeighbor=0;

    } // if (hotclient_makeDiagnostics_)

  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient CLEANUP -> "<<cpu_timer.cpuTime()<<endl;
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

  if ( debug_>1 ) cout << "HcalHotCellClient: report" << endl;
  this->setup();

  ostringstream name;
  name<<process_.c_str()<<"Hcal/HotCellMonitor_Hcal/Hot Cell Task Event Number";
  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_>1 ) cout << "Found '" << name.str().c_str() << "'" << endl;
  }
  getHistograms();
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient REPORT -> "<<cpu_timer.cpuTime()<<endl;
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
  // dummy histograms
  TH2F* dummy2D = new TH2F();
  TH1F* dummy1D = new TH1F();

  // Set Problem cell palette (green = 0 = good, red = 1 = bad)


  // Grab individual histograms
  name<<process_.c_str()<<"HotCellMonitor_Hcal/ ProblemHotCells";
  ProblemHotCells = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");

  getSJ6histos("HotCellMonitor_Hcal/problem_hotcells/", " Problem Hot Cell Rate", ProblemHotCellsByDepth);

  if (hotclient_test_persistent_) getSJ6histos("HotCellMonitor_Hcal/hot_rechit_always_above_threshold/",   "Hot Cells Persistently Above Energy Threshold", AbovePersistentThresholdCellsByDepth);
  if (hotclient_test_pedestal_)  getSJ6histos("HotCellMonitor_Hcal/hot_pedestaltest/", "Hot Cells Above Pedestal", AbovePedestalHotCellsByDepth);
  if (hotclient_test_neighbor_)  getSJ6histos("HotCellMonitor_Hcal/hot_neighbortest/", "Hot Cells Failing Neighbor Test", AboveNeighborsHotCellsByDepth);
  if (hotclient_test_energy_)    getSJ6histos("HotCellMonitor_Hcal/hot_rechit_above_threshold/",   "Hot Cells Above Energy Threshold", AboveEnergyThresholdCellsByDepth);

  if (hotclient_makeDiagnostics_)
    {
      getSJ6histos("HotCellMonitor_Hcal/diagnostics/rechitenergy/","Average rec hit energy per cell",d_avgrechitenergymap);
      d_HBnormped=getAnyHisto(dummy1D,(process_+"HotCellMonitor_Hcal/diagnostics/pedestal/HB_normped").c_str(), process_, dbe_, debug_, cloneME_);
      d_HBrechitenergy=getAnyHisto(dummy1D,(process_+"HotCellMonitor_Hcal/diagnostics/rechitenergy/HB_rechitenergy").c_str(), process_, dbe_, debug_, cloneME_);
      d_HBenergyVsNeighbor=getAnyHisto(dummy2D,(process_+"HotCellMonitor_Hcal/diagnostics/neighborcells/HB_energyVsNeighbor").c_str(), process_, dbe_, debug_, cloneME_);
      d_HEnormped=getAnyHisto(dummy1D,(process_+"HotCellMonitor_Hcal/diagnostics/pedestal/HE_normped").c_str(), process_, dbe_, debug_, cloneME_);
      d_HErechitenergy=getAnyHisto(dummy1D,(process_+"HotCellMonitor_Hcal/diagnostics/rechitenergy/HE_rechitenergy").c_str(), process_, dbe_, debug_, cloneME_);
      d_HEenergyVsNeighbor=getAnyHisto(dummy2D,(process_+"HotCellMonitor_Hcal/diagnostics/neighborcells/HE_energyVsNeighbor").c_str(), process_, dbe_, debug_, cloneME_);
      d_HOnormped=getAnyHisto(dummy1D,(process_+"HotCellMonitor_Hcal/diagnostics/pedestal/HO_normped").c_str(), process_, dbe_, debug_, cloneME_);
      d_HOrechitenergy=getAnyHisto(dummy1D,(process_+"HotCellMonitor_Hcal/diagnostics/rechitenergy/HO_rechitenergy").c_str(), process_, dbe_, debug_, cloneME_);
      d_HOenergyVsNeighbor=getAnyHisto(dummy2D,(process_+"HotCellMonitor_Hcal/diagnostics/neighborcells/HO_energyVsNeighbor").c_str(), process_, dbe_, debug_, cloneME_);
      d_HFnormped=getAnyHisto(dummy1D,(process_+"HotCellMonitor_Hcal/diagnostics/pedestal/HF_normped").c_str(), process_, dbe_, debug_, cloneME_);
      d_HFrechitenergy=getAnyHisto(dummy1D,(process_+"HotCellMonitor_Hcal/diagnostics/rechitenergy/HF_rechitenergy").c_str(), process_, dbe_, debug_, cloneME_);
      d_HFenergyVsNeighbor=getAnyHisto(dummy2D,(process_+"HotCellMonitor_Hcal/diagnostics/neighborcells/HF_energyVsNeighbor").c_str(), process_, dbe_, debug_, cloneME_);
    } // if (hotclient_makeDiagnostics_)


  // Force min/max on problemcells
  for (int i=0;i<6;++i)
    {
      if (ProblemHotCellsByDepth[i])
	{
	  ProblemHotCellsByDepth[i]->SetMaximum(1);
	  ProblemHotCellsByDepth[i]->SetMinimum(0);
	}
      name.str("");

    } // for (int i=0;i<6;++i)
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient GETHISTOGRAMS -> "<<cpu_timer.cpuTime()<<endl;
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
      if ( debug_>1 ) cout << "<HcalHotCellClient::analyze>  Running analyze "<<endl;
    }
  //getHistograms(); // unnecessary, I think
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient ANALYZE -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalHotCellClient::analyze(void)


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

  // Reset individual histograms
  name<<process_.c_str()<<"HotCellMonitor_Hcal/ ProblemHotCells";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  for (int i=0;i<6;++i)
    {
      // Reset arrays of histograms
      // Problem Pedestal Plots
      name<<process_.c_str()<<"HotCellMonitor_Hcal/problem_hotcells/"<<subdets_[i]<<" Problem Hot Cell Rate";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      if (hotclient_test_persistent_)
	{
	  name<<process_.c_str()<<"HotCellMonitor_Hcal/hot_unoccupied_digi/"<<subdets_[i]<<"Hot Cells with No Digis";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (hotclient_test_pedestal_)
	{
	  name<<process_.c_str()<<"HotCellMonitor_Hcal/hot_pedestaltest"<<subdets_[i]<<"Hot Cells Failing Pedestal Test";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (hotclient_test_neighbor_)
	{
	  name<<process_.c_str()<<"HotCellMonitor_Hcal/hot_neighbortest"<<subdets_[i]<<"Hot Cells Failing Neighbor Test";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (hotclient_test_energy_)
	{
	  name<<process_.c_str()<<"HotCellMonitor_Hcal/hot_energytest"<<subdets_[i]<<"Hot Cells Failing Energy Threshold Test";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (hotclient_makeDiagnostics_)
	{
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/pedestal/HB_normped").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/rechitenergy/HB_rechitenergy").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/neighborcells/HB_energyVsNeighbor").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/pedestal/HE_normped").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/rechitenergy/HE_rechitenergy").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/neighborcells/HE_energyVsNeighbor").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/pedestal/HO_normped").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/rechitenergy/HO_rechitenergy").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/neighborcells/HO_energyVsNeighbor").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/pedestal/HF_normped").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/rechitenergy/HF_rechitenergy").c_str(),dbe_);
	  resetME((process_+"HotCellMonitor_Hcal/diagnostics/neighborcells/HF_energyVsNeighbor").c_str(),dbe_);
	} // if (hotclient_makeDiagnostics_)
      
    } // for (int i=0;i<6;++i)
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient RESETALLME -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalHotCellClient::resetAllME()


void HcalHotCellClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  getHistograms();
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>1) cout << "Preparing HcalHotCellClient html output ..." << endl;

  string client = "HotCellMonitor";

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Hot Cell Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Hot Cells</span></h2> " << endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal Hot Cell Status</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  htmlFile << "</h3>" << endl;

  htmlFile << "<table align=\"center\" border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  htmlAnyHisto(runNo,ProblemHotCells,"i#eta","i#phi", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<endl;
  htmlFile<<"<tr align=\"center\"><td> A cell is considered hot if it meets any of the following criteria:"<<endl;
  if (hotclient_test_persistent_) htmlFile<<"<br> A cell's ADC sum is more than (pedestal + N sigma); "<<endl;
  if (hotclient_test_pedestal_ ) htmlFile<<"<br> A cell's energy is above some threshold value X;"<<endl;
  if (hotclient_test_energy_   ) htmlFile<<"<br> A cell's energy is consistently above some threshold value Y (where Y does not necessarily equal X);"<<endl;
  if (hotclient_test_neighbor_ ) htmlFile<<"<br> A cell's energy is much more than the sum of its neighbors;"<<endl;
  htmlFile<<"</td>"<<endl;
  htmlFile<<"</tr></table>"<<endl;
  htmlFile<<"<hr><table align=\"center\" border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile<<"<tr><td align=center><a href=\"Expert_"<< htmlName<<"\"><h2>Detailed Hot Cell Plots</h2> </a></br></td>"<<endl;
  htmlFile<<"</tr></table><br><hr>"<<endl;
  
  // Now print out problem cells
  htmlFile <<"<br>"<<endl;
  htmlFile << "<h2><strong>Hcal Problem Cells</strong></h2>" << endl;
  htmlFile << "(A problem cell is listed below if its failure rate exceeds "<<(100.*minErrorFlag_)<<"%).<br><br>"<<endl;
  htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile <<"<td> Problem Hot Cells<br>(ieta, iphi, depth)</td><td align=\"center\"> Fraction of Events <br>in which cells are bad (%)</td></tr>"<<endl;

  if (ProblemHotCells==0)
    {
      if (debug_) cout <<"<HcalHotCellClient::htmlOutput>  ERROR: can't find Problem Hot Cell plot!"<<endl;
      return;
    }
  int etabins  = ProblemHotCells->GetNbinsX();
  int phibins  = ProblemHotCells->GetNbinsY();
  float etaMin = ProblemHotCells->GetXaxis()->GetXmin();
  float phiMin = ProblemHotCells->GetYaxis()->GetXmin();

  int eta,phi;

  ostringstream name;
  for (int depth=0;depth<6; ++depth)
    {
      for (int ieta=1;ieta<=etabins;++ieta)
        {
          for (int iphi=1; iphi<=phibins;++iphi)
            {
              eta=ieta+int(etaMin)-1;
              phi=iphi+int(phiMin)-1;
	      if (abs(eta)>20 && phi%2!=1) continue;
	      if (abs(eta)>39 && phi%4!=3) continue;
	      int mydepth=depth+1;
	      if (mydepth>4) mydepth-=4; // last two depth values are for HE depth 1,2
	      if (ProblemHotCellsByDepth[depth]==0)
		{
		  continue;
		}
	      if (ProblemHotCellsByDepth[depth]->GetBinContent(ieta,iphi)>minErrorFlag_)
		{
		  if (depth<2)
		    (fabs(eta)<29) ? name<<"HB" : name<<"HF";
		  else if (depth==3)
		    (fabs(eta)<42) ? name<<"HO" : name<<"ZDC";
		  else name <<"HE";
		  htmlFile<<"<td>"<<name.str().c_str()<<" ("<<eta<<", "<<phi<<", "<<mydepth<<")</td><td align=\"center\">"<<ProblemHotCellsByDepth[depth]->GetBinContent(ieta,iphi)*100.<<"</td></tr>"<<endl;

		  name.str("");
		}
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // for (int depth=0;...)
  
  
  // html page footer
  htmlFile <<"</table> " << endl;
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();
  htmlExpertOutput(runNo, htmlDir, htmlName);

  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient HTMLOUTPUT  -> "<<cpu_timer.cpuTime()<<endl;
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
    cout <<" <HcalHotCellClient::htmlExpertOutput>  Preparing Expert html output ..." <<endl;
  
  string client = "HotCellMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_); // does this do anything?

ofstream htmlFile;
  htmlFile.open((htmlDir +"Expert_"+ htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Hot Cell Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile <<"<a name=\"EXPERT_HOTCELL_TOP\" href = \".\"> Back to Main HCAL DQM Page </a><br>"<<endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Hot Cell Status Page </a><br>"<<endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Hot Cells</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table width=100%  border = 1>"<<endl;
  htmlFile << "<tr><td align=\"center\" colspan=1><a href=\"#OVERALL_PROBLEMS\">PROBLEM CELLS BY DEPTH </a></td></tr>"<<endl;
  htmlFile << "<tr><td align=\"center\">"<<endl;
  if (hotclient_test_pedestal_ ) htmlFile<<"<br><a href=\"#PED_PROBLEMS\">Hot cell according to Pedestal Test </a>"<<endl;
  if (hotclient_test_energy_   ) htmlFile<<"<br><a href=\"#ENERGY_PROBLEMS\">Hot cell according to Energy Threshold Test </a>"<<endl;
  if (hotclient_test_persistent_) htmlFile<<"<br><a href=\"#PERSISTENT_PROBLEMS\">Hot cell consistently above a certain energy </a>"<<endl;
  if (hotclient_test_neighbor_ ) htmlFile<<"<br><a href=\"#NEIGHBOR_PROBLEMS\">Hot cell according to Neighbor Test </a>"<<endl;
  htmlFile << "</td></tr>"<<endl;
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><br>"<<endl;


  // Plot overall errors
  htmlFile << "<h2><strong><a name=\"OVERALL_PROBLEMS\">Eta-Phi Maps of Problem Cells By Depth</strong></h2>"<<endl;
  htmlFile <<" These plots of problem cells combine results from all hot cell tests<br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  
  // Depths are stored as:  0:  HB/HF depth 1, 1:  HB/HF 2, 2:  HE 3, 3:  HO/ZDC, 4: HE 1, 5:  HE2
  // remap so that HE depths are plotted consecutively
  int mydepth[6]={0,1,4,5,2,3};
  for (int i=0;i<3;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,ProblemHotCellsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ProblemHotCellsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }

  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><hr><br>"<<endl;
  
 
  // Hot cells failing pedestal tests
  if (hotclient_test_pedestal_)
    {
      htmlFile << "<h2><strong><a name=\"PED_PROBLEMS\">Pedestal Test Problems</strong></h2>"<<endl;
      htmlFile <<"A cell fails this test if its ADC sum is above (pedestal + Nsigma) for  "<<hotclient_checkNevents_pedestal_<<" consecutive events <br>"<<endl;
      htmlFile <<"<a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top</a><br>"<<endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\"> " << endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<3;++i)
	{
	  htmlFile << "<tr align=\"left\">" << endl;
	  htmlAnyHisto(runNo,AbovePedestalHotCellsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,AbovePedestalHotCellsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<endl;
	}
      if (hotclient_makeDiagnostics_)
	{
	  htmlFile <<"<tr align=\"left\">" <<endl;
	  htmlAnyHisto(runNo, d_HBnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlAnyHisto(runNo, d_HEnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlFile <<"</tr>"<<endl;
	  htmlFile <<"<tr align=\"left\">" <<endl;
	  htmlAnyHisto(runNo, d_HOnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlAnyHisto(runNo, d_HFnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlFile <<"</tr>"<<endl;
	} // if (hotclient_makeDiagnostics_)
      htmlFile <<"</table>"<<endl;
      htmlFile <<"<br><hr><br>"<<endl;
    }

  // Hot cells failing energy tests
  if (hotclient_test_energy_)
    {
      htmlFile << "<h2><strong><a name=\"ENERGY_PROBLEMS\">Energy Threshold Test Problems</strong></h2>"<<endl;
      htmlFile <<"A cell fails this test if its rechit energy is above threshold at any time.<br>"<<endl;
      htmlFile <<"<a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top</a><br>"<<endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\"> " << endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<3;++i)
	{
	  htmlFile << "<tr align=\"left\">" << endl;
	  htmlAnyHisto(runNo,AboveEnergyThresholdCellsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,AboveEnergyThresholdCellsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<endl;
	}
      if (hotclient_makeDiagnostics_)
	{
	  htmlFile <<"<tr align=\"left\">" <<endl;
	  htmlAnyHisto(runNo, d_HBrechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlAnyHisto(runNo, d_HErechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlFile <<"</tr>"<<endl;
	  htmlFile <<"<tr align=\"left\">" <<endl;
	  htmlAnyHisto(runNo, d_HOrechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlAnyHisto(runNo, d_HFrechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlFile <<"</tr>"<<endl;
	} // if (hotclient_makeDiagnostics_)

      htmlFile <<"</table>"<<endl;
      htmlFile <<"<br><hr><br>"<<endl;
    }

  // Hot cells persistently above some threshold energy
  if (hotclient_test_persistent_)
    {
      htmlFile << "<h2><strong><a name=\"PERSISTENT_PROBLEMS\">Persistent Hot Cell Problems</strong></h2>"<<endl;
      htmlFile <<"A cell fails this test if its rechit energy is above threshold for "<<hotclient_checkNevents_persistent_<<" consecutive events.<br>"<<endl;
      htmlFile <<"<a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top</a><br>"<<endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\"> " << endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<3;++i)
	{
	  htmlFile << "<tr align=\"left\">" << endl;
	  htmlAnyHisto(runNo,AbovePersistentThresholdCellsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir,0,0);
	  htmlAnyHisto(runNo,AbovePersistentThresholdCellsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir,0,0);
	  htmlFile <<"</tr>"<<endl;
	}
      htmlFile <<"</table>"<<endl;
      htmlFile <<"<br><hr><br>"<<endl;
    }


  // Hot cells failing neighbor tests
  if (hotclient_test_neighbor_)
    {
      htmlFile << "<h2><strong><a name=\"NEIGHBOR_PROBLEMS\">Neighbor Energy Test Problems</strong></h2>"<<endl;
      htmlFile <<"A cell fails this test if its rechit energy is significantly greater than the sum of its surrounding neighbors <br>"<<endl;
      htmlFile <<"<a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top</a><br>"<<endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\"> " << endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<3;++i)
	{
	  htmlFile << "<tr align=\"left\">" << endl;
	  htmlAnyHisto(runNo,AboveNeighborsHotCellsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,AboveNeighborsHotCellsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<endl;
	}
      if (hotclient_makeDiagnostics_)
	{
	  gStyle->SetPalette(1);  // back to rainbow coloring
	  htmlFile <<"<tr align=\"left\">" <<endl;
	  htmlAnyHisto(runNo, d_HBenergyVsNeighbor, "Cell energy (GeV)","Neighbor energy (GeV)", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo, d_HEenergyVsNeighbor, "Cell energy (GeV)","Neighbor energy (GeV)", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<endl;
	  htmlFile <<"<tr align=\"left\">" <<endl;
	  htmlAnyHisto(runNo, d_HOenergyVsNeighbor, "Cell energy (GeV)","Neighbor energy (GeV)", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo, d_HFenergyVsNeighbor, "Cell energy (GeV)","Neighbor energy (GeV)", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<endl;
	} // if (hotclient_makeDiagnostics_)

      htmlFile <<"</table>"<<endl;
      htmlFile <<"<br><hr><br>"<<endl;
    }


  htmlFile <<"<br><hr><br><a href= \"#EXPERT_HOTCELL_TOP\" > Back to Top of Page </a><br>"<<endl;
  htmlFile <<"<a href = \".\"> Back to Main HCAL DQM Page </a><br>"<<endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Hot Cell Status Page </a><br>"<<endl;

  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();

  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient  HTMLEXPERTOUTPUT ->"<<cpu_timer.cpuTime()<<endl;
    }
  return;
} // void HcalHotCellClient::htmlExpertOutput(...)



void HcalHotCellClient::loadHistograms(TFile* infile)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/HotCellMonitor_Hcal/Hot Cell Task Event Number");
  if(tnd)
    {
      string s =tnd->GetTitle();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    }

  ostringstream name;
  // Grab individual histograms
  name<<process_.c_str()<<"HotCellMonitor_Hcal/ ProblemHotCells";
  ProblemHotCells = (TH2F*)infile->Get(name.str().c_str());
  name.str("");
  
  for (int i=0;i<6;++i)
    {
      // Grab arrays of histograms
      name<<process_.c_str()<<"HotCellMonitor_Hcal/problem_pedestals/"<<subdets_[i]<<" Problem Pedestal Rate";
      ProblemHotCellsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      if (hotclient_test_persistent_)
	{
	  name<<process_.c_str()<<"HotCellMonitor_Hcal/hot_unoccupied_digi/"<<subdets_[i]<<"Hot Cells with No Digis";
	  AbovePersistentThresholdCellsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
	  name.str("");
	}
      if (hotclient_test_pedestal_)
	{
	  name<<process_.c_str()<<"HotCellMonitor_Hcal/hot_pedestaltest"<<subdets_[i]<<"Hot Cells Failing Pedestal Test";
	  AbovePedestalHotCellsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
	  name.str("");
	}
      if (hotclient_test_neighbor_)
	{
	  name<<process_.c_str()<<"HotCellMonitor_Hcal/hot_neighbortest"<<subdets_[i]<<"Hot Cells Failing Neighbor Test";
	  AboveNeighborsHotCellsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
	  name.str("");
	}
      if (hotclient_test_energy_)
	{
	  name<<process_.c_str()<<"HotCellMonitor_Hcal/hot_energytest"<<subdets_[i]<<"Hot Cells Failing Energy Threshold Test";
	  AboveEnergyThresholdCellsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
	  name.str("");
	}

    } //for (int i=0;i<6;++i)
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalHotCellClient LOAD HISTOGRAMS -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalHotCellClient::loadHistograms(...)


bool HcalHotCellClient::hasErrors_Temp()
{
  int problemcount=0;

  int etabins  = ProblemHotCells->GetNbinsX();
  int phibins  = ProblemHotCells->GetNbinsY();
  float etaMin = ProblemHotCells->GetXaxis()->GetXmin();
  float phiMin = ProblemHotCells->GetYaxis()->GetXmin();
  int eta,phi;

  for (int depth=0;depth<6; ++depth)
    {
      for (int ieta=1;ieta<=etabins;++ieta)
        {
          for (int iphi=1; iphi<=phibins;++iphi)
            {
              eta=ieta+int(etaMin)-1;
              phi=iphi+int(phiMin)-1;
	      int mydepth=depth+1;
	      if (mydepth>4) mydepth-=4; // last two depth values are for HE depth 1,2
	      if (ProblemHotCellsByDepth[depth]==0)
		{
		  continue;
		}
	      if (ProblemHotCellsByDepth[depth]->GetBinContent(ieta,iphi)>minErrorFlag_)
		{
		  problemcount++;
		}
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // for (int depth=0;...)

  if (problemcount>=100) return true;
  return false;

} // bool HcalHotCellClient::hasErrors_Temp()

bool HcalHotCellClient::hasWarnings_Temp()
{
  int problemcount=0;

  int etabins  = ProblemHotCells->GetNbinsX();
  int phibins  = ProblemHotCells->GetNbinsY();
  float etaMin = ProblemHotCells->GetXaxis()->GetXmin();
  float phiMin = ProblemHotCells->GetYaxis()->GetXmin();
  int eta,phi;
 
  for (int depth=0;depth<6; ++depth)
    {
      for (int ieta=1;ieta<=etabins;++ieta)
        {
          for (int iphi=1; iphi<=phibins;++iphi)
            {
              eta=ieta+int(etaMin)-1;
              phi=iphi+int(phiMin)-1;
	      int mydepth=depth+1;
	      if (mydepth>4) mydepth-=4; // last two depth values are for HE depth 1,2
	      if (ProblemHotCellsByDepth[depth]==0)
		{
		  continue;
		}
	      if (ProblemHotCellsByDepth[depth]->GetBinContent(ieta,iphi)>minErrorFlag_)
		{
		  problemcount++;
		}
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // for (int depth=0;...)

  if (problemcount>0) return true;
  return false;

} // bool HcalHotCellClient::hasWarnings_Temp()
