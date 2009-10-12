#include <DQM/HcalMonitorClient/interface/HcalPedestalClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <math.h>
#include <iostream>

HcalPedestalClient::HcalPedestalClient(){} // constructor 

void HcalPedestalClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  // Get variable values from cfg file
  nominalPedMeanInADC_      = ps.getUntrackedParameter<double>("PedestalClient_nominalPedMeanInADC",3);
  nominalPedWidthInADC_     = ps.getUntrackedParameter<double>("PedestalClient_nominalPedWidthInADC",1);
  maxPedMeanDiffADC_        = ps.getUntrackedParameter<double>("PedestalClient_maxPedMeanDiffADC",1);
  maxPedWidthDiffADC_       = ps.getUntrackedParameter<double>("PedestalClient_maxPedWidthDiffADC",1);
  doFCpeds_                 = ps.getUntrackedParameter<bool>("PedestalClient_pedestalsInFC",1);
  startingTimeSlice_        = ps.getUntrackedParameter<int>("PedestalClient_startingTimeSlice",0);
  endingTimeSlice_          = ps.getUntrackedParameter<int>("PedestalClient_endingTimgSlice",1);
  minErrorFlag_             = ps.getUntrackedParameter<double>("PedestalClient_minErrorFlag",0.05);
  makeDiagnostics_          = ps.getUntrackedParameter<bool>("PedestalMonitor_makeDiagnosticPlots",false);
  // Set individual pointers to NULL
  ProblemPedestals=0;


 // Map the histograms to their names //
        
  // Database Pedestal values
  mapHist2D("BaselineMonitor_Hcal/reference_pedestals/adc/","Pedestal Values from DataBase ADC",ADC_PedestalFromDBByDepth);
  mapHist2D("BaselineMonitor_Hcal/reference_pedestals/adc/","Pedestal Widths from DataBase ADC",ADC_WidthFromDBByDepth);
  mapHist2D("BaselineMonitor_Hcal/reference_pedestals/fc/","Pedestal Values from DataBase fC",fC_PedestalFromDBByDepth);
  mapHist2D("BaselineMonitor_Hcal/reference_pedestals/fc/","Pedestal Widths from DataBase fC",fC_WidthFromDBByDepth);
  mapHist1D("BaselineMonitor_Hcal/reference_pedestals/adc/","1D Reference Pedestal Values ADC",ADC_PedestalFromDBByDepth_1D);
  mapHist1D("BaselineMonitor_Hcal/reference_pedestals/adc/","1D Reference Pedestal Widths ADC",ADC_WidthFromDBByDepth_1D);
  mapHist1D("BaselineMonitor_Hcal/reference_pedestals/fc/","1D Reference Pedestal Values fC",fC_PedestalFromDBByDepth_1D);
  mapHist1D("BaselineMonitor_Hcal/reference_pedestals/fc/","1D Reference Pedestal Widths fC",fC_WidthFromDBByDepth_1D);


  // Raw, sub Pedestals in ADC
  mapHist2D("BaselineMonitor_Hcal/adc/unsubtracted/","Pedestal Values Map ADC", rawADCPedestalMean);
  mapHist2D("BaselineMonitor_Hcal/adc/unsubtracted/","Pedestal Widths Map ADC", rawADCPedestalRMS);
  mapHist2D("BaselineMonitor_Hcal/adc/subtracted(BETA)/","Subtracted Pedestal Values Map ADC", subADCPedestalMean);
  mapHist2D("BaselineMonitor_Hcal/adc/subtracted(BETA)/","Subtracted Pedestal Widths Map ADC", subADCPedestalRMS);
  mapHist1D("BaselineMonitor_Hcal/adc/unsubtracted/","1D Pedestal Values ADC",rawADCPedestalMean_1D);
  mapHist1D("BaselineMonitor_Hcal/adc/unsubtracted/","1D Pedestal Widths ADC",rawADCPedestalRMS_1D);
  mapHist1D("BaselineMonitor_Hcal/adc/subtracted(BETA)/","1D Subtracted Pedestal Values ADC", subADCPedestalMean_1D);
  mapHist1D("BaselineMonitor_Hcal/adc/subtracted(BETA)/","1D Subtracted Pedestal Widths ADC", subADCPedestalRMS_1D);

  // Raw, sub Pedestals in fC
  mapHist2D("BaselineMonitor_Hcal/fc/unsubtracted/","Pedestal Values Map fC", rawfCPedestalMean);
  mapHist2D("BaselineMonitor_Hcal/fc/unsubtracted/","Pedestal Widths Map fC", rawfCPedestalRMS);
  mapHist2D("BaselineMonitor_Hcal/fc/subtracted(BETA)/","Subtracted Pedestal Values Map fC", subfCPedestalMean);
  mapHist2D("BaselineMonitor_Hcal/fc/subtracted(BETA)/","Subtracted Pedestal Widths Map fC", subfCPedestalRMS);
  mapHist1D("BaselineMonitor_Hcal/fc/unsubtracted/","1D Pedestal Values fC",rawfCPedestalMean_1D);
  mapHist1D("BaselineMonitor_Hcal/fc/unsubtracted/","1D Pedestal Widths fC",rawfCPedestalRMS_1D);
  mapHist1D("BaselineMonitor_Hcal/fc/subtracted(BETA)/","1D Subtracted Pedestal Values fC", subfCPedestalMean_1D);
  mapHist1D("BaselineMonitor_Hcal/fc/subtracted(BETA)/","1D Subtracted Pedestal Widths fC", subfCPedestalRMS_1D);

  for (int i=0;i<4;++i)
    {
      // Set each array's pointers to NULL
      // Basic Pedestal plots
      ProblemPedestalsByDepth[i]=0;
      MeanMapByDepth[i]=0;
      RMSMapByDepth[i]=0;

      // Mapped plots
      for( HistMap1D_t::iterator it = histMap1D.begin(); it != histMap1D.end(); ++it )
	it->second.hist[i] = NULL;
      for( HistMap2D_t::iterator it = histMap2D.begin(); it != histMap2D.end(); ++it )
	it->second.hist[i] = NULL;
    }  

  subdets_.push_back("HB HE HF Depth 1 ");
  subdets_.push_back("HB HE HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO ");

  subdets1D_.push_back("HB ");
  subdets1D_.push_back("HE ");
  subdets1D_.push_back("HO ");
  subdets1D_.push_back("HF ");

  return;
} // void HcalPedestalClient::init(...)


HcalPedestalClient::~HcalPedestalClient()
{
  this->cleanup(); // causes crash?
  
} // destructor


void HcalPedestalClient::beginJob(const EventSetup& eventSetup){

  if ( debug_ ) std::cout << "HcalPedestalClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  return;
} // void HcalPedestalClient::beginJob(const EventSetup& eventSetup);


void HcalPedestalClient::beginRun(void)
{
  if ( debug_ ) std::cout << "HcalPedestalClient: beginRun" << std::endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
} // void HcalPedestalClient::beginRun(void)


void HcalPedestalClient::endJob(void) 
{
  if ( debug_ ) std::cout << "HcalPedestalClient: endJob, ievt = " << ievt_ << std::endl;

  //this->cleanup(); // causes crash?
  return;
} // void HcalPedestalClient::endJob(void)


void HcalPedestalClient::endRun(void) 
{
  if ( debug_ ) std::cout << "HcalPedestalClient: endRun, jevt = " << jevt_ << std::endl;
  //this->cleanup(); // causes crash?
  return;
} // void HcalPedestalClient::endRun(void)


void HcalPedestalClient::setup(void) 
{
  return;
} // void HcalPedestalClient::setup(void)


void HcalPedestalClient::cleanup(void) 
{

  // seems to cause crash; leave deletions to framework?

  if(cloneME_)
    {
      // delete individual histogram pointers

      if (ProblemPedestals) delete ProblemPedestals;
      
      for (int i=0;i<4;++i)
	{
	  // delete pointers within arrays of histograms
	  if (ProblemPedestalsByDepth[i]) delete ProblemPedestalsByDepth[i];
	  if (MeanMapByDepth[i]) delete MeanMapByDepth[i];
	  if (RMSMapByDepth[i]) delete RMSMapByDepth[i];

	  for( HistMap1D_t::iterator it = histMap1D.begin(); it != histMap1D.end(); ++it )
	    it->second.hist[i] = NULL;
	  for( HistMap2D_t::iterator it = histMap2D.begin(); it != histMap2D.end(); ++it )
	    it->second.hist[i] = NULL;
	}

    } // if (cloneME)

  // Set individual pointers to NULL

  ProblemPedestals = 0;

  for (int i=0;i<4;++i)
    {
      // Set each array's pointers to NULL
      ProblemPedestalsByDepth[i]=0;
      MeanMapByDepth[i]=0;
      RMSMapByDepth[i]=0;
      ADC_PedestalFromDBByDepth[i]=0;
      ADC_WidthFromDBByDepth[i]=0;
      fC_PedestalFromDBByDepth[i]=0;
      fC_WidthFromDBByDepth[i]=0;
      ADC_PedestalFromDBByDepth_1D[i]=0;
      ADC_WidthFromDBByDepth_1D[i]=0;
      fC_PedestalFromDBByDepth_1D[i]=0;
      fC_WidthFromDBByDepth_1D[i]=0;
      // Raw pedestals in ADC
      rawADCPedestalMean[i]=0;
      rawADCPedestalRMS[i]=0;
      rawADCPedestalMean_1D[i]=0;
      rawADCPedestalRMS_1D[i]=0;
      
      // Raw pedestals in ADC
      subADCPedestalMean[i]=0;
      subADCPedestalRMS[i]=0;
      subADCPedestalMean_1D[i]=0;
      subADCPedestalRMS_1D[i]=0;
      
      // Raw pedestals in FC
      rawfCPedestalMean[i]=0;
      rawfCPedestalRMS[i]=0;
      rawfCPedestalMean_1D[i]=0;
      rawfCPedestalRMS_1D[i]=0;
      
      // Raw pedestals in FC
      subfCPedestalMean[i]=0;
      subfCPedestalRMS[i]=0;
      subfCPedestalMean_1D[i]=0;
      subfCPedestalRMS_1D[i]=0;
    }

  /*
  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  */
  return;
} // void HcalPedestalClient::cleanup(void)


void HcalPedestalClient::report()
{
  if(!dbe_) return;
  if ( debug_ ) std::cout << "HcalPedestalClient: report" << std::endl;
  this->setup();
  getHistograms();
  return;
} // HcalPedestalClient::report()


void HcalPedestalClient::getHistograms()
{
  if(!dbe_) return;

  // Grab individual histograms
  ostringstream name;
  name<<process_.c_str()<<"Hcal/BaselineMonitor_Hcal/Pedestal Task Event Number";
  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_ ) std::cout << "Found '" << name.str().c_str() << "'" << std::endl;
  }
  name.str("");

  name<<process_.c_str()<<"BaselineMonitor_Hcal/ ProblemPedestals";
  ProblemPedestals = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");

  for (int i=0;i<4;++i)
    {
      // Grab arrays of histograms
      getEtaPhiHists("BaselineMonitor_Hcal/problem_pedestals/"," Problem Pedestal Rate", ProblemPedestalsByDepth);

      if (ProblemPedestalsByDepth[i])
	{
	  ProblemPedestalsByDepth[i]->SetMaximum(1);
	  ProblemPedestalsByDepth[i]->SetMinimum(0);
	}


      // Get Overall Pedestal Maps
      getEtaPhiHists("BaselineMonitor_Hcal/","Pedestal Mean Map ADC",MeanMapByDepth);
      if (MeanMapByDepth[i])
	{
	  // This forces green color to get centered at nominal value in colz plots.
	  // It also causes overflow values to get filled at 2*nominalPedMeanInADC_, rather than their true values.
	  // But this is okay -- true values are dumped on main page when listing problem cells.
	  MeanMapByDepth[i]->SetMaximum(2*nominalPedMeanInADC_);
	  MeanMapByDepth[i]->SetMinimum(0);
	}
      getEtaPhiHists("BaselineMonitor_Hcal/","Pedestal RMS Map ADC",RMSMapByDepth);
      if (RMSMapByDepth[i])
	{
	  RMSMapByDepth[i]->SetMaximum(2*nominalPedWidthInADC_ );
	  RMSMapByDepth[i]->SetMinimum(0);
	}
    } // for (int i=0;i<4;++i)
  
  for( HistMap1D_t::iterator it = histMap1D.begin(); it != histMap1D.end(); ++it )
    getSJ6histos(it->second.file, it->second.name, it->second.hist);
  for( HistMap2D_t::iterator it = histMap2D.begin(); it != histMap2D.end(); ++it )
    getEtaPhiHists(it->second.file, it->second.name, it->second.hist);

  return;
} //void HcalPedestalClient::getHistograms()


void HcalPedestalClient::analyze(void)
{
  jevt_++;
  if ( jevt_ % 10 == 0 ) 
    {
      if ( debug_ ) std::cout << "<HcalPedestalClient::analyze>  Running analyze "<<std::endl;
    }
  //getHistograms();
  return;
} // void HcalPedestalClient::analyze(void)


void HcalPedestalClient::createTests()
{
  // Removed a bunch of code that was in older versions of HcalPedestalClient
  // tests should now be handled from outside
  if(!dbe_) return;
  return;
} // void HcalPedestalClient::createTests()


void HcalPedestalClient::resetAllME()
{
  if(!dbe_) return;
  
  ostringstream name;

  // Reset individual histograms
  name<<process_.c_str()<<"BaselineMonitor_Hcal/ ProblemPedestals";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  for (int i=0;i<4;++i)
    {
      // Reset arrays of histograms

      // Problem Pedestal Plots
      name<<process_.c_str()<<"BaselineMonitor_Hcal/problem_pedestals/"<<subdets_[i]<<" Problem Pedestal Rate";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      // Overall Mean Map
      name<<process_.c_str()<<"BaselineMonitor_Hcal/"<<subdets_[i]<<"Pedestal Mean Map ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      
      // Overall Pedestal Map
      name<<process_.c_str()<<"BaselineMonitor_Hcal/"<<subdets_[i]<<"Pedestal RMS Map ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      for( HistMap1D_t::iterator it = histMap1D.begin(); it != histMap1D.end(); ++it ){
	name << process_.c_str() << it->second.file << subdets_[i] << it->second.name;
	resetME(name.str().c_str(), dbe_);
	name.str("");
      }
      for( HistMap2D_t::iterator it = histMap2D.begin(); it != histMap2D.end(); ++it ){
	name << process_.c_str() << it->second.file << subdets_[i] << it->second.name;
	resetME(name.str().c_str(), dbe_);
	name.str("");
      }
    }
} // void HcalPedestalClient::resetAllME()


void HcalPedestalClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_) std::cout << "Preparing HcalPedestalClient html output ..." << std::endl;

  string client = "PedestalMonitor";

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal Pedestal Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Pedestals</span></h2> " << std::endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<h2><strong>Hcal Pedestal Status</strong></h2>" << std::endl;
  htmlFile << "<h3>" << std::endl;
  htmlFile << "</h3>" << std::endl;

  htmlFile << "<table align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  htmlAnyHisto(runNo,ProblemPedestals,"i#eta","i#phi", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile<<"<tr align=\"center\"><td> A pedestal is considered problematic if its mean differs from the nominal mean value of "<<nominalPedMeanInADC_<<" by more than "<<maxPedMeanDiffADC_<<" ADC counts.<br>";
  htmlFile<<"It is also considered problematic if its RMS differs from the nominal RMS of "<<nominalPedWidthInADC_<<" by more than "<<maxPedWidthDiffADC_<<" ADC counts.<br>"<<std::endl;
  htmlFile<<"</tr></table>"<<std::endl;
  htmlFile<<"<hr><table align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile<<"<tr><td align=center><a href=\"Expert_"<< htmlName<<"\"><h2>Detailed Pedestal Plots</h2> </a></br></td>"<<std::endl;
  htmlFile<<"</tr></table><br><hr>"<<std::endl;
  
  // Now print out problem cells
  htmlFile <<"<br>"<<std::endl;
  htmlFile << "<h2><strong>Hcal Problem Cells</strong></h2>" << std::endl;
  htmlFile << "(A problem cell is listed below if its failure rate exceeds "<<(100.*minErrorFlag_)<<"%).<br><br>"<<std::endl;
  htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile <<"<td> Problem Cells<br>(ieta, iphi, depth)</td><td align=\"center\"> Fraction of Events <br>in which cells are bad (%)</td><td align=\"center\"> Mean (ADC)</td><td align=\"center\"> RMS (ADC)</td></tr>"<<std::endl;

  if (ProblemPedestals==0)
    {
      std::cout <<"<HcalPedestalClient::htmlOutput>  ERROR: can't find ProblemPedestal plot!"<<std::endl;
      return;
    }
  
  int ieta=-9999,iphi=-9999;
  int etabins=0, phibins=0;
  ostringstream name;
  for (int depth=0;depth<4; ++depth)
    {
      etabins  = ProblemPedestalsByDepth[depth]->GetNbinsX();
      phibins  = ProblemPedestalsByDepth[depth]->GetNbinsY();
      for (int hist_eta=0;hist_eta<etabins;++hist_eta)
        {
	  ieta=CalcIeta(hist_eta,depth+1);
	  if (ieta==-9999) continue;
          for (int hist_phi=0; hist_phi<phibins;++hist_phi)
            {
              iphi=hist_phi+1;
	      if (abs(ieta)>20 && iphi%2!=1) continue;
	      if (abs(ieta)>39 && iphi%4!=3) continue;
	      
	      if (ProblemPedestalsByDepth[depth]==0)
		  continue;
	      if (ProblemPedestalsByDepth[depth]->GetBinContent(hist_eta+1,hist_phi+1)>minErrorFlag_)
		{
		  if (depth<2)
		    {
		      if (isHB(hist_eta,depth+1)) name <<"HB";
		      else if (isHE(hist_eta,depth+1)) name<<"HE";
		      else if (isHF(hist_eta,depth+1)) name<<"HF";
		    }
		  else if (depth==2) name <<"HE";
		  else if (depth==3) name<<"HO";
		  if (MeanMapByDepth[depth]!=0 && RMSMapByDepth[depth]!=0)
		    htmlFile<<"<td>"<<name.str().c_str()<<" ("<<ieta<<", "<<iphi<<", "<<depth+1<<")</td><td align=\"center\">"<<ProblemPedestalsByDepth[depth]->GetBinContent(hist_eta+1,hist_phi+1)*100.<<"</td><td align=\"center\"> "<<MeanMapByDepth[depth]->GetBinContent(hist_eta+1,hist_phi+1)<<" </td>  <td align=\"center\">"<<RMSMapByDepth[depth]->GetBinContent(hist_eta+1,hist_phi+1)<<"</td></tr>"<<std::endl;
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
      cpu_timer.stop();  std::cout <<"TIMER:: HcalPedestalClient HTMLOUTPUT  -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} //void HcalPedestalClient::htmlOutput(int runNo, ...) 


/////////////////////////////////////////////////////////////////////////////////////

void HcalPedestalClient::htmlExpertOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_) 
    std::cout <<" <HcalPedestalClient::htmlExpertOutput>  Preparing Expert html output ..." <<std::endl;
  
  string client = "PedestalMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_); // does this do anything?

ofstream htmlFile;
  htmlFile.open((htmlDir +"Expert_"+ htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal Pedestal Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile <<"<a name=\"EXPERT_PEDESTAL_TOP\" href = \".\"> Back to Main HCAL DQM Page </a><br>"<<std::endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Pedestal Status Page </a><br>"<<std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Pedestals</span></h2> " << std::endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<table width=100%  border = 1>"<<std::endl;
  htmlFile << "<tr><td align=\"center\" colspan=2><a href=\"#OVERALL_PEDS\">Pedestal Mean and RMS Maps </a></td></tr>"<<std::endl;
  htmlFile << "<tr><td align=\"center\"><a href=\"#RAW_ADC\">Pedestals in ADC counts</a></td>"<<std::endl;
  htmlFile << "<td align=\"center\"><a href=\"#RAW_fC\">Pedestals in femtoCoulombs</a></td></tr>"<<std::endl;
  htmlFile << "<tr><td align=\"center\" colspan=2><a href=\"#PROBLEM_PEDS\">Problem Pedestals</a></td></tr>"<<std::endl;
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><br>"<<std::endl;


  htmlFile << "<h2><strong><a name=\"OVERALL_PEDS\">2D Maps of Pedestal Means and RMS Values</strong></h2>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile <<"These plots show the calculated Mean and RMS of each cell in eta-phi space.<br>"<<std::endl;
  htmlFile <<"These plots are used for determining problem pedestals."<<std::endl;
  htmlFile <<"A pedestal is considered to be a problem if its mean is outside the range "<<nominalPedMeanInADC_<<" +/- "<<maxPedMeanDiffADC_<<" ADC counts, <br>"<<std::endl;
  htmlFile <<"or if its RMS value is outside the range "<<nominalPedWidthInADC_<<" +/- "<<maxPedWidthDiffADC_<<" ADC counts.<br>"<<std::endl;
  // Depths are stored as:  0:  HB/HF depth 1, 1:  HB/HF 2, 2:  HE 3, 3:  HO/ZDC, 4: HE 1, 5:  HE2
  // remap so that HE depths are plotted consecutively

  htmlFile << "<table \align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile <<"<tr><td> Pedestal Mean </td><td> Pedestal RMS </td></tr>"<<std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      MeanMapByDepth[i]->SetMinimum(0);
      MeanMapByDepth[i]->SetMaximum(2*nominalPedMeanInADC_);
      htmlAnyHisto(runNo,MeanMapByDepth[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      RMSMapByDepth[i]->SetMinimum(0);
      RMSMapByDepth[i]->SetMaximum(2*nominalPedMeanInADC_);
      htmlAnyHisto(runNo,RMSMapByDepth[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;
  
  // Plot Pedestal Mean and RMS values
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1); // set back to normal rainbow color
  
  // Pedestals (ADC)
  htmlFile << "<h2><strong><a name=\"RAW_ADC\">Pedestals in ADC</strong></h2>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile <<" Measured Pedestal means come directly from ADC values in cell digis.<br>"<<std::endl;
  htmlFile <<" Reference pedestal values from the database are converted from fC.<br>"<<std::endl;
  htmlFile <<" The conversion of database widths from fC to ADC is still being checked, and may not be correct.<br><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" style = \"width: 100%\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;

  gStyle->SetPalette(1); //set back to normal rainbow color
  htmlFile <<"<tr align=\"center\"><td>Calculated Pedestal Mean</td><td>Mean from Database</td>"<<std::endl;
  htmlFile<<"<td>Subtracted (Calculated-Database)</td></tr>"<<std::endl;
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,rawADCPedestalMean[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ADC_PedestalFromDBByDepth[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subADCPedestalMean[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"<tr align=\"center\"><td>Calculated Pedestal Width</td><td>Width from Database</td>"<<std::endl;
  htmlFile<<"<td>Subtracted (Calculated-Database)</td></tr>"<<std::endl;
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,rawADCPedestalRMS[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ADC_WidthFromDBByDepth[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subADCPedestalRMS[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }

  // Draw 1-D histograms
  htmlFile <<"</table><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile <<"<tr align=\"center\"><td>1D Pedestal Means (calculated)</td><td>1D Pedestal Means (from Database)</td>"<<std::endl;
  htmlFile <<"<td>Subtracted Means (Calculated - Database)</td></tr>"<<std::endl;
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,rawADCPedestalMean_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ADC_PedestalFromDBByDepth_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subADCPedestalMean_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"<tr align=\"center\"><td>1D Pedestal Widths (calculated)</td><td>1D Pedestal Widths (from Database)</td>"<<std::endl;
  htmlFile <<"<td>Subtracted Widths (Calculated - Database)</td></tr>"<<std::endl;
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,rawADCPedestalRMS_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ADC_WidthFromDBByDepth_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subADCPedestalRMS_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;

   // Raw Pedestals (fC)
  htmlFile << "<h2><strong><a name=\"RAW_fC\">Pedestals in femtoCoulombs (fC)</strong></h2>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile <<" Measured Pedestal means and widths are calculated from fC values converted from the ADC counts stored in cell digis.<br>"<<std::endl;
  htmlFile <<" Reference pedestal values are read directly from the database values (in fC).<br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1); //set back to normal rainbow color
  htmlFile <<"<tr align=\"center\"><td>Calculated Pedestal Mean</td><td>Mean from Database</td>"<<std::endl;
  htmlFile<<"<td>Subtracted (Calculated-Database)</td></tr>"<<std::endl;
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,rawfCPedestalMean[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,fC_PedestalFromDBByDepth[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subfCPedestalMean[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"<tr align=\"center\"><td>Calculated Pedestal Width</td><td>Width from Database</td>"<<std::endl;
  htmlFile<<"<td>Subtracted (Calculated-Database)</td></tr>"<<std::endl;
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,rawfCPedestalRMS[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,fC_WidthFromDBByDepth[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subfCPedestalRMS[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }

  // Draw 1-D histograms
  htmlFile <<"</table><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile <<"<tr align=\"center\"><td>1D Pedestal Means (calculated)</td><td>1D Pedestal Means (from Database)</td>"<<std::endl;
  htmlFile <<"<td>Subtracted Means (Calculated - Database)</td></tr>"<<std::endl;
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,rawfCPedestalMean_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,fC_PedestalFromDBByDepth_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subfCPedestalMean_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"<tr align=\"center\"><td>1D Pedestal Widths (calculated)</td><td>1D Pedestal Widths (from Database)</td>"<<std::endl;
  htmlFile <<"<td>Subtracted Widths (Calculated - Database)</td></tr>"<<std::endl;
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,rawfCPedestalRMS_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,fC_WidthFromDBByDepth_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subfCPedestalRMS_1D[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;


  // Plot Pedestal Plot Errors 
  htmlFile << "<h2><strong><a name=\"PROBLEM_PEDS\">Problem Pedestals</strong></h2>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(20,pcol_error_); //set back to normal rainbow color
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,ProblemPedestalsByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ProblemPedestalsByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;

  // html file footer
  htmlFile <<"<br><hr><br><a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top of Page </a><br>"<<std::endl;
  htmlFile <<"<a href = \".\"> Back to Main HCAL DQM Page </a><br>"<<std::endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Pedestal Status Page </a><br>"<<std::endl;

  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  
  htmlFile.close();

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalPedestalClient  HTMLEXPERTOUTPUT ->"<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalPedestalClient::htmlExpertOutput(...)



void HcalPedestalClient::loadHistograms(TFile* infile)
{
  // deprecated function; no longer used
  return;
} // void HcalPedestalClient::loadHistograms(...)


bool HcalPedestalClient::hasErrors_Temp()
{
  int problemcount=0;
  int etabins  = 0;
  int phibins  = 0;
 
  for (int depth=0;depth<4; ++depth)
    {
      etabins=ProblemPedestalsByDepth[depth]->GetNbinsX();
      phibins=ProblemPedestalsByDepth[depth]->GetNbinsY();
      if (ProblemPedestalsByDepth[depth]==0) continue;
      for (int hist_eta=0;hist_eta<etabins;++hist_eta)
        {
          for (int hist_phi=0;hist_phi<phibins;++hist_phi)
            {
	      if (ProblemPedestalsByDepth[depth]->GetBinContent(hist_eta+1,hist_phi+1)>minErrorFlag_)
		{
		  problemcount++;
		}
	    } // for (int hist_phi=0;...)
	} // for (int hist_eta=0;...)
    } // for (int depth=0;...)

  if (problemcount>=1000) return true;
  return false;

} // bool HcalPedestalClient::hasErrors_Temp()

bool HcalPedestalClient::hasWarnings_Temp()
{
  int problemcount=0;
  int etabins  = 0;
  int phibins  = 0;
 
  for (int depth=0;depth<4; ++depth)
    {
      etabins=ProblemPedestalsByDepth[depth]->GetNbinsX();
      phibins=ProblemPedestalsByDepth[depth]->GetNbinsY();
      if (ProblemPedestalsByDepth[depth]==0) continue;
      for (int hist_eta=0;hist_eta<etabins;++hist_eta)
        {
          for (int hist_phi=0;hist_phi<phibins;++hist_phi)
            {
	      if (ProblemPedestalsByDepth[depth]->GetBinContent(hist_eta+1,hist_phi+1)>minErrorFlag_)
		{
		  problemcount++;
		}
	    } // for (int hist_phi=0;...)
	} // for (int hist_eta=0;...)
    } // for (int depth=0;...)

  if (problemcount>0) return true;
  return false;

} // bool HcalPedestalClient::hasWarnings_Temp()
