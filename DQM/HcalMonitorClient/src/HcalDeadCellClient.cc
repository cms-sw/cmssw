#include <DQM/HcalMonitorClient/interface/HcalDeadCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <math.h>
#include <iostream>

#define BITSHIFT 5

HcalDeadCellClient::HcalDeadCellClient(){} // constructor 

void HcalDeadCellClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  // Get variable values from cfg file
  // Set which dead cell checks will looked at
  deadclient_test_occupancy_         = ps.getUntrackedParameter<bool>("DeadCellClient_test_occupancy",true);
  deadclient_test_rechit_occupancy_  = ps.getUntrackedParameter<bool>("DeadCellClient_test_occupancy",true);
  deadclient_test_pedestal_          = ps.getUntrackedParameter<bool>("DeadCellClient_test_pedestal",true);
  deadclient_test_neighbor_          = ps.getUntrackedParameter<bool>("DeadCellClient_test_neighbor",false);
  deadclient_test_energy_            = ps.getUntrackedParameter<bool>("DeadCellClient_test_energy",true);

  deadclient_checkNevents_ = ps.getUntrackedParameter<int>("DeadCellClient_checkNevents",100);

  minErrorFlag_ = ps.getUntrackedParameter<double>("DeadCellClient_minErrorFlag",0.0);

  deadclient_makeDiagnostics_ = ps.getUntrackedParameter<bool>("DeadCellClient_makeDiagnosticPlots",false);

  dump2database_ = false; // eventually make this configurable

  // Set histograms to NULL
  ProblemDeadCells=0;
  for (int i=0;i<6;++i)
    {
      // Set each array's pointers to NULL
      ProblemDeadCellsByDepth[i]=0;
      if (deadclient_test_occupancy_) UnoccupiedDeadCellsByDepth[i]=0;
      if (deadclient_test_rechit_occupancy_) UnoccupiedRecHitsByDepth[i]=0;
      if (deadclient_test_pedestal_) BelowPedestalDeadCellsByDepth[i]=0;
      if (deadclient_test_neighbor_) BelowNeighborsDeadCellsByDepth[i]=0;
      if (deadclient_test_energy_) BelowEnergyThresholdCellsByDepth[i]=0;
    }  

  if (deadclient_makeDiagnostics_)
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
    } // if (deadclient_makeDiagnostics_)

  subdets_.push_back("HB HF Depth 1 ");
  subdets_.push_back("HB HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO ZDC ");
  subdets_.push_back("HE Depth 1 ");
  subdets_.push_back("HE Depth 2 ");

  return;
} // void HcalDeadCellClient::init(...)


HcalDeadCellClient::~HcalDeadCellClient()
{
  this->cleanup();
} // destructor


void HcalDeadCellClient::beginJob(const EventSetup& eventSetup){

  if ( debug_>1 ) std::cout << "HcalDeadCellClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  return;
} // void HcalDeadCellClient::beginJob(const EventSetup& eventSetup);


void HcalDeadCellClient::beginRun(void)
{
  if ( debug_>1 ) std::cout << "HcalDeadCellClient: beginRun" << std::endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
} // void HcalDeadCellClient::beginRun(void)


void HcalDeadCellClient::endJob(std::map<HcalDetId, unsigned int>& myqual) 
{
  if ( debug_>1 ) std::cout << "HcalDeadCellClient: endJob, ievt = " << ievt_ << std::endl;

  if (dump2database_==true) // don't do anything special unless specifically asked to dump db file
    {
      int eta,phi;
      float binval;
      int mydepth;

      int subdet;
      char* subdetname;
      if (debug_>1)
	{
	  std::cout <<"<HcalDeadCellClient>  Summary of Dead Cells in Run: "<<std::endl;
	  std::cout <<"(Error rate must be >= "<<minErrorFlag_*100.<<"% )"<<std::endl;  
	}

      float etaMin = ProblemDeadCells->GetXaxis()->GetXmin();
      float phiMin = ProblemDeadCells->GetYaxis()->GetXmin();
      int etabins  = ProblemDeadCells->GetNbinsX();
      int phibins  = ProblemDeadCells->GetNbinsY();
      for (int ieta=1;ieta<=etabins;++ieta)
	{
	  for (int iphi=1;iphi<=phibins;++iphi)
	    {
	      eta=ieta+int(etaMin)-1;
	      phi=iphi+int(phiMin)-1;
	      
	      for (int d=0;d<6;++d)
		{
		  // ProblemDeadCells have already been normalized
		  binval=ProblemDeadCellsByDepth[d]->GetBinContent(ieta,iphi);
		  
		  // Set subdetector labels for output
		  if (d<2) // HB/HF
		    {
		      if (abs(eta)<29)
			{
			  subdetname="HB";
			  subdet=1;
			}
		      else
			{
			  subdetname="HF";
			  subdet=4;
			}
		    }
		  else if (d==3)
		    {
		      if (abs(eta)==43)
			{
			  subdetname="ZDC";
			  subdet=7; // correct value??
			}
		      else
			{
			  subdetname="HO";
			  subdet=3;
			}
		    }
		  else
		    {
		      subdetname="HE";
		      subdet=2;
		    }
		  // Set correct depth label
		  if (d>3)
		    mydepth=d-3;
		  else
		    mydepth=d+1;
		  HcalDetId myid((HcalSubdetector)(subdet), eta, phi, mydepth);
		  // Need this to keep from flagging non-existent HE/HF cells
		  if (!validDetId((HcalSubdetector)(subdet), eta, phi, mydepth))
		    continue;
		  if (binval<=minErrorFlag_)
		    continue;
		  if (debug_>0)
		    std::cout <<"Hot Cell "<<subdet<<"("<<eta<<", "<<phi<<", "<<mydepth<<"):  "<<binval*100.<<"%"<<std::endl;

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
		} // for (int d=0;d<6;++d) // loop over depth histograms
	    } // for (int iphi=1;iphi<=phibins;++iphi)
	} // for (int ieta=1;ieta<=etabins;++ieta)
    } // if (dump2database_==true)



  this->cleanup();
  return;
} // void HcalDeadCellClient::endJob(void)


void HcalDeadCellClient::endRun(void) 
{
  if ( debug_>1 ) std::cout << "HcalDeadCellClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();
  return;
} // void HcalDeadCellClient::endRun(void)


void HcalDeadCellClient::setup(void) 
{
  return;
} // void HcalDeadCellClient::setup(void)


void HcalDeadCellClient::cleanup(void) 
{
  // leave deletions to framework
  if(1<0 && cloneME_)
    {
      // delete individual histogram pointers
      if (ProblemDeadCells) delete ProblemDeadCells;

      for (int i=0;i<6;++i)
	{
	  // delete pointers within arrays of histograms
	  if (ProblemDeadCellsByDepth[i])           delete ProblemDeadCellsByDepth[i];
	  if (UnoccupiedDeadCellsByDepth[i])        delete UnoccupiedDeadCellsByDepth[i];
	  if (UnoccupiedRecHitsByDepth[i])          delete UnoccupiedRecHitsByDepth[i];
	  if (BelowPedestalDeadCellsByDepth[i])     delete BelowPedestalDeadCellsByDepth[i];
	  if (BelowNeighborsDeadCellsByDepth[i])    delete BelowNeighborsDeadCellsByDepth[i];
	  if (BelowEnergyThresholdCellsByDepth[i])  delete BelowEnergyThresholdCellsByDepth[i];
	}
      
      if (deadclient_makeDiagnostics_)
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
	} // if (deadclient_makeDiagnostics_)
      

    }

  // Set individual pointers to NULL
  /*
  ProblemDeadCells = 0;

  for (int i=0;i<6;++i)
    {
      // Set each array's pointers to NULL
      ProblemDeadCellsByDepth[i]=0;
      UnoccupiedDeadCellsByDepth[i]=0;
      UnoccupiedRecHitsByDepth[i]=0;
      BelowPedestalDeadCellsByDepth[i]=0;
      BelowNeighborsDeadCellsByDepth[i]=0;
      BelowEnergyThresholdCellsByDepth[i]=0;
    }
  
  if (deadclient_makeDiagnostics_)
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
    } // if (deadclient_makeDiagnostics_)
  */

  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  return;
} // void HcalDeadCellClient::cleanup(void)


void HcalDeadCellClient::report()
{
  if(!dbe_) return;
  if ( debug_>1 ) std::cout << "HcalDeadCellClient: report" << std::endl;
  this->setup();

  getHistograms();

  return;
} // HcalDeadCellClient::report()


void HcalDeadCellClient::getHistograms()
{
  if(!dbe_) return;



  ostringstream name;
  name<<process_.c_str()<<"Hcal/DeadCellMonitor_Hcal/Dead Cell Task Event Number";
  // Get ievt_ value
  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_>1 ) std::cout << "Found '" << name.str().c_str() << "'" << std::endl;
  }


  // dummy histograms
  TH2F* dummy2D = new TH2F();
  TH1F* dummy1D = new TH1F();

  // Set Problem cell palette (green = 0 = good, red = 1 = bad)


  // Grab individual histograms
  name.str("");
  name<<process_.c_str()<<"DeadCellMonitor_Hcal/ ProblemDeadCells";
  ProblemDeadCells = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");

  getSJ6histos("DeadCellMonitor_Hcal/problem_deadcells/", " Problem Dead Cell Rate", ProblemDeadCellsByDepth);
  if (deadclient_test_occupancy_) getSJ6histos("DeadCellMonitor_Hcal/dead_unoccupied_digi/",   "Dead Cells with No Digis", UnoccupiedDeadCellsByDepth);
  if (deadclient_test_rechit_occupancy_) getSJ6histos("DeadCellMonitor_Hcal/dead_unoccupied_rechit/",   "Dead Cells with No Rec Hits", UnoccupiedRecHitsByDepth);
 
  if (deadclient_test_pedestal_)  getSJ6histos("DeadCellMonitor_Hcal/dead_pedestaltest/", "Dead Cells Failing Pedestal Test", BelowPedestalDeadCellsByDepth);
  if (deadclient_test_neighbor_)  getSJ6histos("DeadCellMonitor_Hcal/dead_neighbortest/", "Dead Cells Failing Neighbor Test", BelowNeighborsDeadCellsByDepth);
  if (deadclient_test_energy_)    getSJ6histos("DeadCellMonitor_Hcal/dead_energytest/",   "Dead Cells Failing Energy Threshold Test", BelowEnergyThresholdCellsByDepth);

  if (deadclient_makeDiagnostics_)
    {
      d_HBnormped=getAnyHisto(dummy1D,(process_+"DeadCellMonitor_Hcal/diagnostics/pedestal/HB_normped").c_str(), process_, dbe_, debug_, cloneME_);
      d_HBrechitenergy=getAnyHisto(dummy1D,(process_+"DeadCellMonitor_Hcal/diagnostics/energythreshold/HB_rechitenergy").c_str(), process_, dbe_, debug_, cloneME_);
      d_HBenergyVsNeighbor=getAnyHisto(dummy2D,(process_+"DeadCellMonitor_Hcal/diagnostics/neighborcells/HB_energyVsNeighbor").c_str(), process_, dbe_, debug_, cloneME_);
      d_HEnormped=getAnyHisto(dummy1D,(process_+"DeadCellMonitor_Hcal/diagnostics/pedestal/HE_normped").c_str(), process_, dbe_, debug_, cloneME_);
      d_HErechitenergy=getAnyHisto(dummy1D,(process_+"DeadCellMonitor_Hcal/diagnostics/energythreshold/HE_rechitenergy").c_str(), process_, dbe_, debug_, cloneME_);
      d_HEenergyVsNeighbor=getAnyHisto(dummy2D,(process_+"DeadCellMonitor_Hcal/diagnostics/neighborcells/HE_energyVsNeighbor").c_str(), process_, dbe_, debug_, cloneME_);
      d_HOnormped=getAnyHisto(dummy1D,(process_+"DeadCellMonitor_Hcal/diagnostics/pedestal/HO_normped").c_str(), process_, dbe_, debug_, cloneME_);
      d_HOrechitenergy=getAnyHisto(dummy1D,(process_+"DeadCellMonitor_Hcal/diagnostics/energythreshold/HO_rechitenergy").c_str(), process_, dbe_, debug_, cloneME_);
      d_HOenergyVsNeighbor=getAnyHisto(dummy2D,(process_+"DeadCellMonitor_Hcal/diagnostics/neighborcells/HO_energyVsNeighbor").c_str(), process_, dbe_, debug_, cloneME_);
      d_HFnormped=getAnyHisto(dummy1D,(process_+"DeadCellMonitor_Hcal/diagnostics/pedestal/HF_normped").c_str(), process_, dbe_, debug_, cloneME_);
      d_HFrechitenergy=getAnyHisto(dummy1D,(process_+"DeadCellMonitor_Hcal/diagnostics/energythreshold/HF_rechitenergy").c_str(), process_, dbe_, debug_, cloneME_);
      d_HFenergyVsNeighbor=getAnyHisto(dummy2D,(process_+"DeadCellMonitor_Hcal/diagnostics/neighborcells/HF_energyVsNeighbor").c_str(), process_, dbe_, debug_, cloneME_);
    } // if (deadclient_makeDiagnostics_)

  // Scale rate histograms
  if (ievt_>0)
    {
      if (ProblemDeadCells && ievt_>0)
	{
	  ProblemDeadCells->Scale(1./ievt_);
	  ProblemDeadCells->SetMaximum(1);
	  ProblemDeadCells->SetMinimum(0);
	}

      // Force min/max on problemcells
      for (int i=0;i<6;++i)
	{
	  if (ievt_>0 && ProblemDeadCellsByDepth[i])
	    {
	      ProblemDeadCellsByDepth[i]->Scale(1./ievt_);
	      ProblemDeadCellsByDepth[i]->SetMaximum(1);
	      ProblemDeadCellsByDepth[i]->SetMinimum(0);
	    }
	}
    } // for (int i=0;i<6;++i)

  return;
} //void HcalDeadCellClient::getHistograms()


void HcalDeadCellClient::analyze(void)
{
  jevt_++;
  if ( jevt_ % 10 == 0 ) 
    {
      if ( debug_>1 ) std::cout << "<HcalDeadCellClient::analyze>  Running analyze "<<std::endl;
    }
  getHistograms(); // not needed here?
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
  
  ostringstream name;

  // Reset individual histograms
  name<<process_.c_str()<<"DeadCellMonitor_Hcal/ ProblemDeadCells";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  for (int i=0;i<6;++i)
    {
      // Reset arrays of histograms
      // Problem Pedestal Plots
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/problem_deadcells/"<<subdets_[i]<<" Problem Dead Cell Rate";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      if (deadclient_test_occupancy_)
	{
	  name<<process_.c_str()<<"DeadCellMonitor_Hcal/dead_unoccupied_digi/"<<subdets_[i]<<"Dead Cells with No Digis";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (deadclient_test_rechit_occupancy_)
	{
	  name<<process_.c_str()<<"DeadCellMonitor_Hcal/dead_unoccupied_rechit/"<<subdets_[i]<<"Dead Cells with No Rec Hits";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}

      if (deadclient_test_pedestal_)
	{
	  name<<process_.c_str()<<"DeadCellMonitor_Hcal/dead_pedestaltest"<<subdets_[i]<<"Dead Cells Failing Pedestal Test";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (deadclient_test_neighbor_)
	{
	  name<<process_.c_str()<<"DeadCellMonitor_Hcal/dead_neighbortest"<<subdets_[i]<<"Dead Cells Failing Neighbor Test";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (deadclient_test_energy_)
	{
	  name<<process_.c_str()<<"DeadCellMonitor_Hcal/dead_energytest"<<subdets_[i]<<"Dead Cells Failing Energy Threshold Test";
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
      if (deadclient_makeDiagnostics_)
	{
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/pedestal/HB_normped").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/energythreshold/HB_rechitenergy").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/neighborcells/HB_energyVsNeighbor").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/pedestal/HE_normped").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/energythreshold/HE_rechitenergy").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/neighborcells/HE_energyVsNeighbor").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/pedestal/HO_normped").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/energythreshold/HO_rechitenergy").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/neighborcells/HO_energyVsNeighbor").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/pedestal/HF_normped").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/energythreshold/HF_rechitenergy").c_str(),dbe_);
	  resetME((process_+"DeadCellMonitor_Hcal/diagnostics/neighborcells/HF_energyVsNeighbor").c_str(),dbe_);
    } // if (deadclient_makeDiagnostics_)

    }
  return;
} // void HcalDeadCellClient::resetAllME()


void HcalDeadCellClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  getHistograms(); 
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
  htmlAnyHisto(runNo,ProblemDeadCells,"i#eta","i#phi", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile<<"<tr align=\"center\"><td> A cell is considered dead if it meets any of the following criteria:"<<std::endl;
  if (deadclient_test_occupancy_) htmlFile<<"<br> A cell's digi is not present for a number of consecutive events; "<<std::endl;
  if (deadclient_test_rechit_occupancy_) htmlFile<<"<br> A cell's rec hit is not present for a number of consecutive events; "<<std::endl;

  if (deadclient_test_pedestal_ ) htmlFile<<"<br> A cell's ADC sum is consistently less than (pedestal + N sigma);"<<std::endl;
  if (deadclient_test_energy_   ) htmlFile<<"<br> A cell's energy is consistently less than a threshold value;"<<std::endl;
  if (deadclient_test_neighbor_ ) htmlFile<<"<br> A cell's energy is much less than the average of its neighbors;"<<std::endl;
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

  if (ProblemDeadCells==0)
    {
      if (debug_) std::cout <<"<HcalDeadCellClient::htmlOutput>  ERROR: can't find Problem Dead Cell plot!"<<std::endl;
      return;
    }
  int etabins  = ProblemDeadCells->GetNbinsX();
  int phibins  = ProblemDeadCells->GetNbinsY();
  float etaMin = ProblemDeadCells->GetXaxis()->GetXmin();
  float phiMin = ProblemDeadCells->GetYaxis()->GetXmin();

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
	      if (ProblemDeadCellsByDepth[depth]==0)
		{
		  continue;
		}
	      if (ProblemDeadCellsByDepth[depth]->GetBinContent(ieta,iphi)>minErrorFlag_)
		{
		  if (depth<2)
		    (fabs(eta)<29) ? name<<"HB" : name<<"HF";
		  else if (depth==3)
		    (fabs(eta)<42) ? name<<"HO" : name<<"ZDC";
		  else name <<"HE";
		  htmlFile<<"<td>"<<name.str().c_str()<<" ("<<eta<<", "<<phi<<", "<<mydepth<<")</td><td align=\"center\">"<<ProblemDeadCellsByDepth[depth]->GetBinContent(ieta,iphi)*100.<<"</td></tr>"<<std::endl;

		  name.str("");
		}
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
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
  if (deadclient_test_occupancy_) htmlFile<<"<br><a href=\"#OCC_PROBLEMS\">Dead cell according to Digi Occupancy Test </a>"<<std::endl;
  if (deadclient_test_rechit_occupancy_) htmlFile<<"<br><a href=\"#OCCRECHIT_PROBLEMS\">Dead cell according to RecHit Occupancy Test </a>"<<std::endl;
  if (deadclient_test_pedestal_ ) htmlFile<<"<br><a href=\"#PED_PROBLEMS\">Dead cell according to Pedestal Test </a>"<<std::endl;
  if (deadclient_test_energy_   ) htmlFile<<"<br><a href=\"#ENERGY_PROBLEMS\">Dead cell according to Energy Threshold Test </a>"<<std::endl;
  if (deadclient_test_neighbor_ ) htmlFile<<"<br><a href=\"#NEIGHBOR_PROBLEMS\">Dead cell according to Neighbor Test </a>"<<std::endl;
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
  
  // Depths are stored as:  0:  HB/HF depth 1, 1:  HB/HF 2, 2:  HE 3, 3:  HO/ZDC, 4: HE 1, 5:  HE2
  // remap so that HE depths are plotted consecutively
  int mydepth[6]={0,1,4,5,2,3};
  for (int i=0;i<3;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,ProblemDeadCellsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ProblemDeadCellsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }

  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;
  
  // Dead cells failing digi occupancy tests
  if (deadclient_test_occupancy_)
    {
      htmlFile << "<h2><strong><a name=\"OCC_PROBLEMS\">Digi Occupancy Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its digi is absent for "<<deadclient_checkNevents_<<" consecutive events<br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_DEADCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<3;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,UnoccupiedDeadCellsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,UnoccupiedDeadCellsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	}
      htmlFile <<"</table>"<<std::endl;
      htmlFile <<"<br><hr><br>"<<std::endl;
    }

  // Dead cells failing rec hit occupancy tests
  if (deadclient_test_rechit_occupancy_)
    {
      htmlFile << "<h2><strong><a name=\"OCCRECHIT_PROBLEMS\">rec HitOccupancy Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its rechit is absent for "<<deadclient_checkNevents_<<" consecutive events<br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_DEADCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<3;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,UnoccupiedRecHitsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,UnoccupiedRecHitsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	}
      htmlFile <<"</table>"<<std::endl;
      htmlFile <<"<br><hr><br>"<<std::endl;
    }

  // Dead cells failing pedestal tests
  if (deadclient_test_pedestal_)
    {
      htmlFile << "<h2><strong><a name=\"PED_PROBLEMS\">Pedestal Test Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its ADC sum is below (pedestal + Nsigma) for  "<<deadclient_checkNevents_<<" consecutive events <br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_DEADCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<3;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,BelowPedestalDeadCellsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,BelowPedestalDeadCellsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	}
      if (deadclient_makeDiagnostics_)
	{
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HBnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlAnyHisto(runNo, d_HEnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlFile <<"</tr>"<<std::endl;
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HOnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlAnyHisto(runNo, d_HFnormped, "(ADC-ped)/width","", 92, htmlFile, htmlDir,1);
	  htmlFile <<"</tr>"<<std::endl;
	} // if (deadclient_makeDiagnostics_)
      htmlFile <<"</table>"<<std::endl;
      htmlFile <<"<br><hr><br>"<<std::endl;
    }

  // Dead cells failing energy tests
  if (deadclient_test_energy_)
    {
      htmlFile << "<h2><strong><a name=\"ENERGY_PROBLEMS\">Energy Threshold Test Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its rechit energy is below threshold for "<<deadclient_checkNevents_<<" consecutive events <br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_DEADCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<3;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,BelowEnergyThresholdCellsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,BelowEnergyThresholdCellsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	}
      if (deadclient_makeDiagnostics_)
	{
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HBrechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlAnyHisto(runNo, d_HErechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlFile <<"</tr>"<<std::endl;
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HOrechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlAnyHisto(runNo, d_HFrechitenergy, "Energy (GeV)","", 92, htmlFile, htmlDir,1,1);
	  htmlFile <<"</tr>"<<std::endl;
	} // if (deadclient_makeDiagnostics_)

      htmlFile <<"</table>"<<std::endl;
      htmlFile <<"<br><hr><br>"<<std::endl;
    }

  // Dead cells failing neighbor tests
  if (deadclient_test_neighbor_)
    {
      htmlFile << "<h2><strong><a name=\"NEIGHBOR_PROBLEMS\">Neighbor Energy Test Problems</strong></h2>"<<std::endl;
      htmlFile <<"A cell fails this test if its rechit energy is significantly less than the average of its surrounding neighbors <br>"<<std::endl;
      htmlFile <<"<a href= \"#EXPERT_DEADCELL_TOP\" > Back to Top</a><br>"<<std::endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
      for (int i=0;i<3;++i)
	{
	  htmlFile << "<tr align=\"left\">" << std::endl;
	  htmlAnyHisto(runNo,BelowNeighborsDeadCellsByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,BelowNeighborsDeadCellsByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	}
      if (deadclient_makeDiagnostics_)
	{
	  gStyle->SetPalette(1);  // back to rainbow coloring
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HBenergyVsNeighbor, "i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo, d_HEenergyVsNeighbor, "i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	  htmlFile <<"<tr align=\"left\">" <<std::endl;
	  htmlAnyHisto(runNo, d_HOenergyVsNeighbor, "i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo, d_HFenergyVsNeighbor, "i#eta","i#phi", 92, htmlFile, htmlDir);
	  htmlFile <<"</tr>"<<std::endl;
	} // if (deadclient_makeDiagnostics_)

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
  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/DeadCellMonitor_Hcal/Dead Cell Task Event Number");
  if(tnd)
    {
      string s =tnd->GetTitle();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    }

  ostringstream name;
  // Grab individual histograms
  name<<process_.c_str()<<"DeadCellMonitor_Hcal/ ProblemDeadCells";
  ProblemDeadCells = (TH2F*)infile->Get(name.str().c_str());
  name.str("");
  
  for (int i=0;i<6;++i)
    {
      // Grab arrays of histograms
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/problem_deadcells/"<<subdets_[i]<<" Problem Dead Cell Rate";
      ProblemDeadCellsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      if (deadclient_test_occupancy_)
	{
	  name<<process_.c_str()<<"DeadCellMonitor_Hcal/dead_unoccupied_digi/"<<subdets_[i]<<"Dead Cells with No Digis";
	  UnoccupiedDeadCellsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
	  name.str("");
	}
      if (deadclient_test_rechit_occupancy_)
	{
	  name<<process_.c_str()<<"DeadCellMonitor_Hcal/dead_unoccupied_rechit/"<<subdets_[i]<<"Dead Cells with No Rec Hits";
	  UnoccupiedRecHitsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
	  name.str("");
	}

      if (deadclient_test_pedestal_)
	{
	  name<<process_.c_str()<<"DeadCellMonitor_Hcal/dead_pedestaltest"<<subdets_[i]<<"Dead Cells Failing Pedestal Test";
	  BelowPedestalDeadCellsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
	  name.str("");
	}
      if (deadclient_test_neighbor_)
	{
	  name<<process_.c_str()<<"DeadCellMonitor_Hcal/dead_neighbortest"<<subdets_[i]<<"Dead Cells Failing Neighbor Test";
	  BelowNeighborsDeadCellsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
	  name.str("");
	}
      if (deadclient_test_energy_)
	{
	  name<<process_.c_str()<<"DeadCellMonitor_Hcal/dead_energytest"<<subdets_[i]<<"Dead Cells Failing Energy Threshold Test";
	  BelowEnergyThresholdCellsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
	  name.str("");
	}

    } //for (int i=0;i<6;++i)

  if (deadclient_makeDiagnostics_)
    {
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HB_normped";
      d_HBnormped = (TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HB_rechitenergy";
      d_HBrechitenergy = (TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HB_energyVsNeighbor";
      d_HBenergyVsNeighbor = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HE_normped";
      d_HEnormped = (TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HE_rechitenergy";
      d_HErechitenergy = (TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HE_energyVsNeighbor";
      d_HEenergyVsNeighbor = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HO_normped";
      d_HOnormped = (TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HO_rechitenergy";
      d_HOrechitenergy = (TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HO_energyVsNeighbor";
      d_HOenergyVsNeighbor = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HF_normped";
      d_HFnormped = (TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HF_rechitenergy";
      d_HFrechitenergy = (TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"DeadCellMonitor_Hcal/diagnostics/pedestal/HF_energyVsNeighbor";
      d_HFenergyVsNeighbor = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
    } // if (deadclient_makeDiagnostics_)



  // Scale rate histograms
  if (ievt_>0)
    {
      ProblemDeadCells->Scale(1./ievt_);
      for (int i=0;i<6;++i)
	ProblemDeadCellsByDepth[i]->Scale(1./ievt_);
      // scale other histograms here as well?
    }

  // Force min/max on problemcells
  for (int i=0;i<6;++i)
    {
      if (ProblemDeadCellsByDepth[i])
	{
	  ProblemDeadCellsByDepth[i]->SetMaximum(1);
	  ProblemDeadCellsByDepth[i]->SetMinimum(0);
	}
      name.str("");

    } // for (int i=0;i<6;++i)

  return;
} // void HcalDeadCellClient::loadHistograms(...)




bool HcalDeadCellClient::hasErrors_Temp()
{
  int problemcount=0;

  int etabins  = ProblemDeadCells->GetNbinsX();
  int phibins  = ProblemDeadCells->GetNbinsY();
  float etaMin = ProblemDeadCells->GetXaxis()->GetXmin();
  float phiMin = ProblemDeadCells->GetYaxis()->GetXmin();
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
	      if (ProblemDeadCellsByDepth[depth]==0)
		{
		  continue;
		}
	      if (ProblemDeadCellsByDepth[depth]->GetBinContent(ieta,iphi)>minErrorFlag_)
		{
		  problemcount++;
		}
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // for (int depth=0;...)

  if (problemcount>=100) return true;
  return false;

} // bool HcalDeadCellClient::hasErrors_Temp()

bool HcalDeadCellClient::hasWarnings_Temp()
{
  int problemcount=0;

  int etabins  = ProblemDeadCells->GetNbinsX();
  int phibins  = ProblemDeadCells->GetNbinsY();
  float etaMin = ProblemDeadCells->GetXaxis()->GetXmin();
  float phiMin = ProblemDeadCells->GetYaxis()->GetXmin();
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
	      if (ProblemDeadCellsByDepth[depth]==0)
		{
		  continue;
		}
	      if (ProblemDeadCellsByDepth[depth]->GetBinContent(ieta,iphi)>minErrorFlag_)
		{
		  problemcount++;
		}
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // for (int depth=0;...)

  if (problemcount>0) return true;
  return false;

} // bool HcalDeadCellClient::hasWarnings_Temp()
