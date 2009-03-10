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

  // Set individual pointers to NULL
  ProblemPedestals=0;

  for (int i=0;i<6;++i)
    {
      // Set each array's pointers to NULL
      ProblemPedestalsByDepth[i]=0;
      MeanMapByDepth[i]=0;
      RMSMapByDepth[i]=0;

      // Basic Pedestal plots
      ADC_PedestalFromDBByDepth[i]=0;
      ADC_WidthFromDBByDepth[i]=0;
      fC_PedestalFromDBByDepth[i]=0;
      fC_WidthFromDBByDepth[i]=0;

      // Pedestals from Database
      ADC_PedestalFromDBByDepth[i]=0;
      ADC_WidthFromDBByDepth[i]=0;
      fC_PedestalFromDBByDepth[i]=0;
      fC_WidthFromDBByDepth[i]=0;

      // Raw pedestals in ADC
      rawADCPedestalMean[i]=0;
      rawADCPedestalRMS[i]=0;
      rawADCPedestalMean_1D[i]=0;
      rawADCPedestalRMS_1D[i]=0;
      
      // Subtracted pedestals in ADC
      subADCPedestalMean[i]=0;
      subADCPedestalRMS[i]=0;
      subADCPedestalMean_1D[i]=0;
      subADCPedestalRMS_1D[i]=0;
      
      // Raw pedestals in FC
      rawFCPedestalMean[i]=0;
      rawFCPedestalRMS[i]=0;
      rawFCPedestalMean_1D[i]=0;
      rawFCPedestalRMS_1D[i]=0;
      
      // Subtracted pedestals in FC
      subFCPedestalMean[i]=0;
      subFCPedestalRMS[i]=0;
      subFCPedestalMean_1D[i]=0;
      subFCPedestalRMS_1D[i]=0;
    }  

  subdets_.push_back("HB HF Depth 1 ");
  subdets_.push_back("HB HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO ZDC ");
  subdets_.push_back("HE Depth 1 ");
  subdets_.push_back("HE Depth 2 ");


  return;
} // void HcalPedestalClient::init(...)


HcalPedestalClient::~HcalPedestalClient()
{
  this->cleanup();
} // destructor


void HcalPedestalClient::beginJob(const EventSetup& eventSetup){

  if ( debug_ ) cout << "HcalPedestalClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  return;
} // void HcalPedestalClient::beginJob(const EventSetup& eventSetup);


void HcalPedestalClient::beginRun(void)
{
  if ( debug_ ) cout << "HcalPedestalClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
} // void HcalPedestalClient::beginRun(void)


void HcalPedestalClient::endJob(void) 
{
  if ( debug_ ) cout << "HcalPedestalClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();
  return;
} // void HcalPedestalClient::endJob(void)


void HcalPedestalClient::endRun(void) 
{
  if ( debug_ ) cout << "HcalPedestalClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();
  return;
} // void HcalPedestalClient::endRun(void)


void HcalPedestalClient::setup(void) 
{
  return;
} // void HcalPedestalClient::setup(void)


void HcalPedestalClient::cleanup(void) 
{
  if(cloneME_)
    {
      // delete individual histogram pointers
      if (ProblemPedestals) delete ProblemPedestals;
      
      for (int i=0;i<6;++i)
	{
	  // delete pointers within arrays of histograms
	  if (ProblemPedestalsByDepth[i]) delete ProblemPedestalsByDepth[i];
	  if (MeanMapByDepth[i]) delete MeanMapByDepth[i];
	  if (RMSMapByDepth[i]) delete RMSMapByDepth[i];

	  // Pedestals from Database
	  if (ADC_PedestalFromDBByDepth[i]) delete ADC_PedestalFromDBByDepth[i];
	  if (ADC_WidthFromDBByDepth[i]) delete ADC_WidthFromDBByDepth[i];
	  if (fC_PedestalFromDBByDepth[i]) delete fC_PedestalFromDBByDepth[i];
	  if (fC_WidthFromDBByDepth[i]) delete fC_WidthFromDBByDepth[i];
	  
	  // Raw pedestals in ADC
	  if (rawADCPedestalMean[i]) delete rawADCPedestalMean[i];
	  if (rawADCPedestalRMS[i]) delete rawADCPedestalRMS[i];
	  if (rawADCPedestalMean_1D[i]) delete rawADCPedestalMean_1D[i];
	  if (rawADCPedestalRMS_1D[i]) delete rawADCPedestalRMS_1D[i];
	  
	  // Subtracted pedestals in ADC
	  if (subADCPedestalMean[i]) delete subADCPedestalMean[i];
	  if (subADCPedestalRMS[i]) delete subADCPedestalRMS[i];
	  if (subADCPedestalMean_1D[i]) delete subADCPedestalMean_1D[i];
	  if (subADCPedestalRMS_1D[i]) delete subADCPedestalRMS_1D[i];
  
	  // Raw pedestals in FC
	  if (rawFCPedestalMean[i]) delete rawFCPedestalMean[i];
	  if (rawFCPedestalRMS[i]) delete rawFCPedestalRMS[i];
	  if (rawFCPedestalMean_1D[i]) delete rawFCPedestalMean_1D[i];
	  if (rawFCPedestalRMS_1D[i]) delete rawFCPedestalRMS_1D[i];
	  
	  // Subtracted pedestals in FC
	  if (subFCPedestalMean[i]) delete subFCPedestalMean[i];
	  if (subFCPedestalRMS[i]) delete subFCPedestalRMS[i];
	  if (subFCPedestalMean_1D[i]) delete subFCPedestalMean_1D[i];
	  if (subFCPedestalRMS_1D[i]) delete subFCPedestalRMS_1D[i];
	}
    }

  // Set individual pointers to NULL
  ProblemPedestals = 0;

  for (int i=0;i<6;++i)
    {
      // Set each array's pointers to NULL
      ProblemPedestalsByDepth[i]=0;
      MeanMapByDepth[i]=0;
      RMSMapByDepth[i]=0;
      ADC_PedestalFromDBByDepth[i]=0;
      ADC_WidthFromDBByDepth[i]=0;
      fC_PedestalFromDBByDepth[i]=0;
      fC_WidthFromDBByDepth[i]=0;
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
      rawFCPedestalMean[i]=0;
      rawFCPedestalRMS[i]=0;
      rawFCPedestalMean_1D[i]=0;
      rawFCPedestalRMS_1D[i]=0;
      
      // Raw pedestals in FC
      subFCPedestalMean[i]=0;
      subFCPedestalRMS[i]=0;
      subFCPedestalMean_1D[i]=0;
      subFCPedestalRMS_1D[i]=0;
    }

  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  return;
} // void HcalPedestalClient::cleanup(void)


void HcalPedestalClient::report()
{
  if(!dbe_) return;
  if ( debug_ ) cout << "HcalPedestalClient: report" << endl;
  this->setup();

  ostringstream name;
  name<<process_.c_str()<<"Hcal/PedestalMonitor_Hcal/Pedestal Task Event Number";
  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_ ) cout << "Found '" << name.str().c_str() << "'" << endl;
  }
  getHistograms();

  return;
} // HcalPedestalClient::report()


void HcalPedestalClient::getHistograms()
{
  if(!dbe_) return;

  // Grab individual histograms
  ostringstream name;
  TH2F* dummy2D = new TH2F();
  name<<process_.c_str()<<"PedestalMonitor_Hcal/ ProblemPedestals";
  ProblemPedestals = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");

  for (int i=0;i<6;++i)
    {
      // Grab arrays of histograms
      getSJ6histos("PedestalMonitor_Hcal/problem_pedestals/"," Problem Pedestal Rate", ProblemPedestalsByDepth);

      if (ProblemPedestalsByDepth[i])
	{
	  ProblemPedestalsByDepth[i]->SetMaximum(1);
	  ProblemPedestalsByDepth[i]->SetMinimum(0);
	}


      // Get Overall Pedestal Maps
      getSJ6histos("PedestalMonitor_Hcal/","Pedestal Mean Map ADC",MeanMapByDepth);
      if (MeanMapByDepth[i])
	{
	  // This forces green color to get centered at nominal value in colz plots.
	  // It also causes overflow values to get filled at 2*nominalPedMeanInADC_, rather than their true values.
	  // But this is okay -- true values are dumped on main page when listing problem cells.
	  MeanMapByDepth[i]->SetMaximum(2*nominalPedMeanInADC_);
	  MeanMapByDepth[i]->SetMinimum(0);
	}
      getSJ6histos("PedestalMonitor_Hcal/","Pedestal RMS Map ADC",RMSMapByDepth);
      if (RMSMapByDepth[i])
	{
	  RMSMapByDepth[i]->SetMaximum(2*nominalPedWidthInADC_ );
	  RMSMapByDepth[i]->SetMinimum(0);
	}
      
      // Database Pedestal values
      getSJ6histos("PedestalMonitor_Hcal/reference_pedestals/adc/","Pedestal Values from DataBase ADC",ADC_PedestalFromDBByDepth);
      getSJ6histos("PedestalMonitor_Hcal/reference_pedestals/adc/","Pedestal Widths from DataBase ADC",ADC_WidthFromDBByDepth);
      getSJ6histos("PedestalMonitor_Hcal/reference_pedestals/fc/","Pedestal Values from DataBase fC",fC_PedestalFromDBByDepth);
      getSJ6histos("PedestalMonitor_Hcal/reference_pedestals/fc/","Pedestal Widths from DataBase fC",fC_WidthFromDBByDepth);


      // Raw, sub Pedestals in ADC
      getSJ6histos("PedestalMonitor_Hcal/adc/raw/","Pedestal Values Map ADC", rawADCPedestalMean);
      getSJ6histos("PedestalMonitor_Hcal/adc/raw/","Pedestal Widths Map ADC", rawADCPedestalRMS);
      getSJ6histos("PedestalMonitor_Hcal/adc/subtracted__beta_testing/","Subtracted Pedestal Values Map ADC", subADCPedestalMean);
      getSJ6histos("PedestalMonitor_Hcal/adc/subtracted__beta_testing/","Subtracted Pedestal Widths Map ADC", subADCPedestalRMS);
      getSJ6histos("PedestalMonitor_Hcal/adc/raw/","1D Pedestal Values ADC",rawADCPedestalMean_1D);
      getSJ6histos("PedestalMonitor_Hcal/adc/raw/","1D Pedestal Widths ADC",rawADCPedestalRMS_1D);
      getSJ6histos("PedestalMonitor_Hcal/adc/subtracted__beta_testing/","1D Subtracted Pedestal Values ADC", subADCPedestalMean_1D);
      getSJ6histos("PedestalMonitor_Hcal/adc/subtracted__beta_testing/","1D Subtracted Pedestal Widths ADC", subADCPedestalRMS_1D);

      // Raw, sub Pedestals in fC
      getSJ6histos("PedestalMonitor_Hcal/fc/raw/","Pedestal Values Map fC", rawFCPedestalMean);
      getSJ6histos("PedestalMonitor_Hcal/fc/raw/","Pedestal Widths Map fC", rawFCPedestalRMS);
      getSJ6histos("PedestalMonitor_Hcal/fc/subtracted__beta_testing/","Subtracted Pedestal Values Map fC", subFCPedestalMean);
      getSJ6histos("PedestalMonitor_Hcal/fc/subtracted__beta_testing/","Subtracted Pedestal Widths Map fC", subFCPedestalRMS);
      getSJ6histos("PedestalMonitor_Hcal/fc/raw/","1D Pedestal Values fC",rawFCPedestalMean_1D);
      getSJ6histos("PedestalMonitor_Hcal/fc/raw/","1D Pedestal Widths fC",rawFCPedestalRMS_1D);
      getSJ6histos("PedestalMonitor_Hcal/fc/subtracted__beta_testing/","1D Subtracted Pedestal Values fC", subFCPedestalMean_1D);
      getSJ6histos("PedestalMonitor_Hcal/fc/subtracted__beta_testing/","1D Subtracted Pedestal Widths fC", subFCPedestalRMS_1D);

    } // for (int i=0;i<6;++i)

  return;
} //void HcalPedestalClient::getHistograms()


void HcalPedestalClient::analyze(void)
{
  jevt_++;
  if ( jevt_ % 10 == 0 ) 
    {
      if ( debug_ ) cout << "<HcalPedestalClient::analyze>  Running analyze "<<endl;
    }
  getHistograms();
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
  name<<process_.c_str()<<"PedestalMonitor_Hcal/ ProblemPedestals";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  for (int i=0;i<6;++i)
    {
      // Reset arrays of histograms

      // Problem Pedestal Plots
      name<<process_.c_str()<<"PedestalMonitor_Hcal/problem_pedestals/"<<subdets_[i]<<" Problem Pedestal Rate";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      // Overall Mean Map
      name<<process_.c_str()<<"PedestalMonitor_Hcal/"<<subdets_[i]<<"Pedestal Mean Map ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      
      // Overall Pedestal Map
      name<<process_.c_str()<<"PedestalMonitor_Hcal/"<<subdets_[i]<<"Pedestal RMS Map ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");


      // Database Pedestal values
      name<<process_.c_str()<<"PedestalMonitor_Hcal/reference_pedestals/adc/"<<subdets_[i]<<"Pedestal Values from DataBase ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      name<<process_.c_str()<<"PedestalMonitor_Hcal/reference_pedestals/adc/"<<subdets_[i]<<"Pedestal Widths from DataBase ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      name<<process_.c_str()<<"PedestalMonitor_Hcal/reference_pedestals/fc/"<<subdets_[i]<<"Pedestal Values from DataBase fC";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      name<<process_.c_str()<<"PedestalMonitor_Hcal/reference_pedestals/fc/"<<subdets_[i]<<"Pedestal Widths from DataBase fC";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      // Raw, sub Pedestals in ADC
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/raw/"<<subdets_[i]<<"Pedestal Values Map ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/raw/"<<subdets_[i]<<"Pedestal Widths Map ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/subtracted__beta_testing/"<<subdets_[i]<<"Subtracted Pedestal Values Map ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/subtracted__beta_testing/"<<subdets_[i]<<"Subtracted Pedestal Widths Map ADC";
  
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/raw/"<<subdets_[i]<<"1D Pedestal Values ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/raw/"<<subdets_[i]<<"1D Pedestal Widths ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/subtracted__beta_testing/"<<subdets_[i]<<"1D Subtracted Pedestal Values ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/subtracted__beta_testing/"<<subdets_[i]<<"1D Subtracted Pedestal Widths ADC";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      // Raw, sub Pedestals in fC
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/raw/"<<subdets_[i]<<"Pedestal Values Map fC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/raw/"<<subdets_[i]<<"Pedestal Widths Map fC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/subtracted__beta_testing/"<<subdets_[i]<<"Subtracted Pedestal Values Map fC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/subtracted__beta_testing/"<<subdets_[i]<<"Subtracted Pedestal Widths Map fC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/raw/"<<subdets_[i]<<"1D Pedestal Values fC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/raw/"<<subdets_[i]<<"1D Pedestal Widths fC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/subtracted__beta_testing/"<<subdets_[i]<<"1D Subtracted Pedestal Values fC";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/subtracted__beta_testing/"<<subdets_[i]<<"1D Subtracted Pedestal Widths fC";
      resetME(name.str().c_str(),dbe_);
      name.str("");

    }
  return;
} // void HcalPedestalClient::resetAllME()


void HcalPedestalClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_) cout << "Preparing HcalPedestalClient html output ..." << endl;

  string client = "PedestalMonitor";

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Pedestal Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Pedestals</span></h2> " << endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal Pedestal Status</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  htmlFile << "</h3>" << endl;

  htmlFile << "<table align=\"center\" border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  htmlAnyHisto(runNo,ProblemPedestals,"i#eta","i#phi", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<endl;
  htmlFile<<"<tr align=\"center\"><td> A pedestal is considered problematic if its mean differs from the nominal mean value of "<<nominalPedMeanInADC_<<" by more than "<<maxPedMeanDiffADC_<<" ADC counts.<br>";
  htmlFile<<"It is also considered problematic if its RMS differs from the nominal RMS of "<<nominalPedWidthInADC_<<" by more than "<<maxPedWidthDiffADC_<<" ADC counts.<br>"<<endl;
  htmlFile<<"</tr></table>"<<endl;
  htmlFile<<"<hr><table align=\"center\" border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile<<"<tr><td align=center><a href=\"Expert_"<< htmlName<<"\"><h2>Detailed Pedestal Plots</h2> </a></br></td>"<<endl;
  htmlFile<<"</tr></table><br><hr>"<<endl;
  
  // Now print out problem cells
  htmlFile <<"<br>"<<endl;
  htmlFile << "<h2><strong>Hcal Problem Cells</strong></h2>" << endl;
  htmlFile << "(A problem cell is listed below if its failure rate exceeds "<<(100.*minErrorFlag_)<<"%).<br><br>"<<endl;
  htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile <<"<td> Problem Cells<br>(ieta, iphi, depth)</td><td align=\"center\"> Fraction of Events <br>in which cells are bad (%)</td><td align=\"center\"> Mean (ADC)</td><td align=\"center\"> RMS (ADC)</td></tr>"<<endl;

  if (ProblemPedestals==0)
    {
      cout <<"<HcalPedestalClient::htmlOutput>  ERROR: can't find ProblemPedestal plot!"<<endl;
      return;
    }
  int etabins  = ProblemPedestals->GetNbinsX();
  int phibins  = ProblemPedestals->GetNbinsY();
  float etaMin = ProblemPedestals->GetXaxis()->GetXmin();
  float phiMin = ProblemPedestals->GetYaxis()->GetXmin();

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
	      int mydepth=depth+1;
	      if (mydepth>4) mydepth-=4; // last two depth values are for HE depth 1,2
	      if (ProblemPedestalsByDepth[depth]==0)
		{
		  continue;
		}
	      if (ProblemPedestalsByDepth[depth]->GetBinContent(ieta,iphi)>minErrorFlag_)
		{
		  if (depth<2)
		    (fabs(eta)<29) ? name<<"HB" : name<<"HF";
		  else if (depth==3)
		    (fabs(eta)<42) ? name<<"HO" : name<<"ZDC";
		  else name <<"HE";
		  if (MeanMapByDepth[depth]!=0 && RMSMapByDepth[depth]!=0)
		    htmlFile<<"<td>"<<name.str().c_str()<<" ("<<eta<<", "<<phi<<", "<<mydepth<<")</td><td align=\"center\">"<<ProblemPedestalsByDepth[depth]->GetBinContent(ieta,iphi)*100.<<"</td><td align=\"center\"> "<<MeanMapByDepth[depth]->GetBinContent(ieta,iphi)<<" </td>  <td align=\"center\">"<<RMSMapByDepth[depth]->GetBinContent(ieta,iphi)<<"</td></tr>"<<endl;
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
      cpu_timer.stop();  cout <<"TIMER:: HcalPedestalClient HTMLOUTPUT  -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} //void HcalPedestalClient::htmlOutput(int runNo, ...) 


void HcalPedestalClient::htmlExpertOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_) 
    cout <<" <HcalPedestalClient::htmlExpertOutput>  Preparing Expert html output ..." <<endl;
  
  string client = "PedestalMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_); // does this do anything?

ofstream htmlFile;
  htmlFile.open((htmlDir +"Expert_"+ htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Pedestal Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile <<"<a name=\"EXPERT_PEDESTAL_TOP\" href = \".\"> Back to Main HCAL DQM Page </a><br>"<<endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Pedestal Status Page </a><br>"<<endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Pedestals</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table width=100%  border = 1>"<<endl;
  htmlFile << "<tr><td align=\"center\" colspan=2><a href=\"#OVERALL_PEDS\">Pedestal Mean and RMS Maps </a></td></tr>"<<endl;
  htmlFile << "<tr><td align=\"center\">Pedestals in ADC counts<br><a href=\"#RAW_ADC\">Raw Pedestals</a><br><a href=\"#SUB_ADC\">Subtracted Pedestals</a><br></td>"<<endl;
  htmlFile << "<td align=\"center\">Pedestals in femtoCoulombs<br><a href=\"#RAW_fC\">Raw Pedestals</a><br><a href=\"#SUB_fC\">Subtracted Pedestals</a><br></td></tr>"<<endl;
  htmlFile << "<tr><td align=\"center\" colspan=2><a href=\"#PROBLEM_PEDS\">Problem Pedestals</a></td></tr>"<<endl;
  htmlFile << "<td align=\"center\"><a href=\"#REF_ADC\">Reference Pedestals <br>from database (ADC)</a><br></td><td align=\"center\"><a href=\"#REF_fC\"> Reference Pedestals <br>from database (fC)</a><br></td></tr>"<<endl;
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><br>"<<endl;


  // Plot Pedestal Mean and RMS values
  htmlFile << "<h2><strong><a name=\"OVERALL_PEDS\">2D Maps of Pedestal Means and RMS Values</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1); // set back to normal rainbow color
  

  // Depths are stored as:  0:  HB/HF depth 1, 1:  HB/HF 2, 2:  HE 3, 3:  HO/ZDC, 4: HE 1, 5:  HE2
  // remap so that HE depths are plotted consecutively
  int mydepth[6]={0,1,4,5,2,3};
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,MeanMapByDepth[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,RMSMapByDepth[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><hr><br>"<<endl;
  
  // Raw Pedestals (ADC)
  htmlFile << "<h2><strong><a name=\"RAW_ADC\">Raw Pedestals from database (ADC)</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1); //set back to normal rainbow color
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,rawADCPedestalMean[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,rawADCPedestalRMS[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,rawADCPedestalMean_1D[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,rawADCPedestalRMS_1D[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><hr><br>"<<endl;

  // Subtracted Pedestals (ADC)
  htmlFile << "<h2><strong><a name=\"SUB_ADC\">Subtracted Pedestals from database (ADC)</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile <<"Plots of (computed pedestal values) - (reference pedestal values).<br>"<<endl;
  htmlFile <<"(If the database stores values in fC, the subtracted pedestal widths in ADC may not be plotted correctly here.)<br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1); //set back to normal rainbow color
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,subADCPedestalMean[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subADCPedestalRMS[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,subADCPedestalMean_1D[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subADCPedestalRMS_1D[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;

   // Raw Pedestals (fC)
  htmlFile << "<h2><strong><a name=\"RAW_fC\">Raw Pedestals from database (fC)</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1); //set back to normal rainbow color
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,rawFCPedestalMean[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,rawFCPedestalRMS[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,rawFCPedestalMean_1D[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,rawFCPedestalRMS_1D[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><hr><br>"<<endl;

  // Subtracted Pedestals (fC)
  htmlFile << "<h2><strong><a name=\"SUB_fC\">Subtracted Pedestals from database (fC)</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<endl; 
  htmlFile <<"Plots of (computed pedestal values) - (reference pedestal values).<br>"<<endl;
  htmlFile <<"(If the database stores values in ADC, the subtracted pedestal widths in fC may not be plotted correctly here.)<br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1); //set back to normal rainbow color
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,subFCPedestalMean[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subFCPedestalRMS[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,subFCPedestalMean_1D[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,subFCPedestalRMS_1D[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;


  // Plot Pedestal Plot Errors 
  htmlFile << "<h2><strong><a name=\"PROBLEM_PEDS\">Problem Pedestals</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(20,pcol_error_); //set back to normal rainbow color
  for (int i=0;i<3;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,ProblemPedestalsByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ProblemPedestalsByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><hr><br>"<<endl;

  //Plot Reference Pedestals
  // Reference Pedestals in ADC
  htmlFile << "<h2><strong><a name=\"REF_ADC\">Reference Pedestals from database (ADC)</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile <<"(If the database stores values in fC, the pedestal widths in ADC may not be plotted correctly here.)<br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1); //set back to normal rainbow color
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,ADC_PedestalFromDBByDepth[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ADC_WidthFromDBByDepth[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;

  htmlFile <<"<br><hr><br>"<<endl;

  // Reference Pedestals in fC
  htmlFile << "<h2><strong><a name=\"REF_fC\">Reference Pedestals from database (fC)</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile <<"(If the database stores values in ADC, the pedestal widths in fC may not be plotted correctly here.)<br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1); //set back to normal rainbow color
  for (int i=0;i<6;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,fC_PedestalFromDBByDepth[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,fC_WidthFromDBByDepth[mydepth[i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;

  htmlFile <<"<br><hr><br><a href= \"#EXPERT_PEDESTAL_TOP\" > Back to Top of Page </a><br>"<<endl;
  htmlFile <<"<a href = \".\"> Back to Main HCAL DQM Page </a><br>"<<endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Pedestal Status Page </a><br>"<<endl;

  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();

  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalPedestalClient  HTMLEXPERTOUTPUT ->"<<cpu_timer.cpuTime()<<endl;
    }
  return;
} // void HcalPedestalClient::htmlExpertOutput(...)



void HcalPedestalClient::loadHistograms(TFile* infile)
{
  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/PedestalMonitor_Hcal/Pedestal Task Event Number");
  if(tnd)
    {
      string s =tnd->GetTitle();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    }

   ostringstream name;
  // Grab individual histograms
  name<<process_.c_str()<<"PedestalMonitor_Hcal/ ProblemPedestals";
  ProblemPedestals = (TH2F*)infile->Get(name.str().c_str());
  name.str("");
  
  for (int i=0;i<6;++i)
    {
      // Grab arrays of histograms
      name<<process_.c_str()<<"PedestalMonitor_Hcal/problem_pedestals/"<<subdets_[i]<<" Problem Pedestal Rate";
      ProblemPedestalsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");

      // Overall Pedestal Map Plots
      name<<process_.c_str()<<"PedestalMonitor_Hcal/"<<subdets_[i]<<"Pedestal Mean Map ADC";
      MeanMapByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      
      name<<process_.c_str()<<"PedestalMonitor_Hcal/"<<subdets_[i]<<"Pedestal RMS Map ADC";
      RMSMapByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");

      // Database Pedestal values
      name<<process_.c_str()<<"PedestalMonitor_Hcal/reference_pedestals/adc/"<<subdets_[i]<<"Pedestal Values from DataBase ADC";
      ADC_PedestalFromDBByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");

      name<<process_.c_str()<<"PedestalMonitor_Hcal/reference_pedestals/adc/"<<subdets_[i]<<"Pedestal Widths from DataBase ADC";
      ADC_WidthFromDBByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");

      name<<process_.c_str()<<"PedestalMonitor_Hcal/reference_pedestals/fc/"<<subdets_[i]<<"Pedestal Values from DataBase fC";
      fC_PedestalFromDBByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");

      name<<process_.c_str()<<"PedestalMonitor_Hcal/reference_pedestals/fc/"<<subdets_[i]<<"Pedestal Widths from DataBase fC";
      fC_WidthFromDBByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");

      // Raw, sub Pedestals in ADC
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/raw/"<<subdets_[i]<<"Pedestal Values Map ADC";
      rawADCPedestalMean[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/raw/"<<subdets_[i]<<"Pedestal Widths Map ADC";
      rawADCPedestalRMS[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/subtracted__beta_testing/"<<subdets_[i]<<"Subtracted Pedestal Values Map ADC";
      subADCPedestalMean[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/subtracted__beta_testing/"<<subdets_[i]<<"Subtracted Pedestal Widths Map ADC";
      subADCPedestalRMS[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/raw/"<<subdets_[i]<<"1D Pedestal Values ADC";
      rawADCPedestalMean_1D[i] =(TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/raw/"<<subdets_[i]<<"1D Pedestal Widths ADC";
      rawADCPedestalRMS_1D[i] =(TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/subtracted__beta_testing/"<<subdets_[i]<<"1D Subtracted Pedestal Values ADC";
      subADCPedestalMean_1D[i] =(TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/adc/subtracted__beta_testing/"<<subdets_[i]<<"1D Subtracted Pedestal Widths ADC";
      subADCPedestalRMS_1D[i] =(TH1F*)infile->Get(name.str().c_str());
      name.str("");

      // Raw, sub Pedestals in fC
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/raw/"<<subdets_[i]<<"Pedestal Values Map fC";
      rawFCPedestalMean[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/raw/"<<subdets_[i]<<"Pedestal Widths Map fC";
      rawFCPedestalRMS[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/subtracted__beta_testing/"<<subdets_[i]<<"Subtracted Pedestal Values Map fC";
      subFCPedestalMean[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/subtracted__beta_testing/"<<subdets_[i]<<"Subtracted Pedestal Widths Map fC";
      subFCPedestalRMS[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/raw/"<<subdets_[i]<<"1D Pedestal Values fC";
      rawFCPedestalMean_1D[i] =(TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/raw/"<<subdets_[i]<<"1D Pedestal Widths fC";
      rawFCPedestalRMS_1D[i] =(TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/subtracted__beta_testing/"<<subdets_[i]<<"1D Subtracted Pedestal Values fC";
      subFCPedestalMean_1D[i] =(TH1F*)infile->Get(name.str().c_str());
      name.str("");
      name<<process_.c_str()<<"PedestalMonitor_Hcal/fc/subtracted__beta_testing/"<<subdets_[i]<<"1D Subtracted Pedestal Widths fC";
      subFCPedestalRMS_1D[i] =(TH1F*)infile->Get(name.str().c_str());
      name.str("");
    } //for (int i=0;i<6;++i)
  return;
} // void HcalPedestalClient::loadHistograms(...)

