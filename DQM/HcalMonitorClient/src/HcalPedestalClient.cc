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

  nominalPedMeanInADC_ = 3;
  nominalPedWidthInADC_ = 1;
  maxPedMeanDiffADC_ = 1;
  maxPedWidthDiffADC_ = 1;
  doFCpeds_ = true;
  startingTimeSlice_ = 0;
  endingTimeSlice_ = 1;

  // Set individual pointers to NULL
  ProblemPedestals=0;

  for (int i=0;i<6;++i)
    {
      // Set each array's pointers to NULL
      ProblemPedestalsByDepth[i]=0;
      MeanMapByDepth[i]=0;
      RMSMapByDepth[i]=0;
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

  /*
  char name[256];    
  sprintf(name, "%sHcal/PedestalMonitor/Pedestal Task Event Number",process_.c_str());
  MonitorElement* me = dbe_->get(name);
  */

  ostringstream name;
  name<<process_.c_str()<<"Hcal/PedestalMonitor/Pedestal Task Event Number";
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
  debug_=true;
  cout <<"HELLO"<<endl;
  ostringstream name;
  // Grab individual histograms
  name<<process_.c_str()<<"Hcal/PedestalMonitor/ ProblemPedestals";
  ProblemPedestals = getHisto2(name.str(),process_,dbe_,debug_,cloneME_);
  name.str("");

  for (int i=0;i<6;++i)
    {
      // Grab arrays of histograms
      name<<process_.c_str()<<"Hcal/PedestalMonitor/problemPedestals/"<<subdets_[i]<<"Problem Pedestal Rate";
      ProblemPedestalsByDepth[i] = getHisto2(name.str(),process_,dbe_,debug_,cloneME_);
      name.str("");

      name<<process_.c_str()<<"Hcal/PedestalMonitor/"<<subdets_[i]<<"Pedestal Mean Map (ADC)";
      MeanMapByDepth[i] = getHisto2(name.str(),process_,dbe_,debug_,cloneME_);
      name.str("");
      
      name<<process_.c_str()<<"Hcal/PedestalMonitor/"<<subdets_[i]<<"Pedestal RMS Map (ADC)";
      RMSMapByDepth[i] = getHisto2(name.str(),process_,dbe_,debug_,cloneME_);
      name.str("");
    }

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

  // Grab individual histograms
  name<<process_.c_str()<<"Hcal/PedestalMonitor/ ProblemPedestals";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  for (int i=0;i<6;++i)
    {
      // Grab arrays of histograms
      name<<process_.c_str()<<"Hcal/PedestalMonitor/problemPedestals/"<<subdets_[i]<<"Problem Pedestal Rate";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      name<<process_.c_str()<<"Hcal/PedestalMonitor/"<<subdets_[i]<<"Pedestal Mean Map (ADC)";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      
      name<<process_.c_str()<<"Hcal/PedestalMonitor/"<<subdets_[i]<<"Pedestal RMS Map (ADC)";
      resetME(name.str().c_str(),dbe_);
      name.str("");
    }
  return;
} // void HcalPedestalClient::resetAllME()


void HcalPedestalClient::htmlOutput(int runNo, string htmlDir, string htmlName){
  
  cout << "Preparing HcalPedestalClient html output ..." << endl;
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
  htmlFile << "<table width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"PedestalMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"PedestalMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"PedestalMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  //htmlFile << "<td><a href=\"badPedestalList.html\">Pedestal Error List</a></td>" << endl;
  htmlFile << "</tr></table>" << endl;

  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal Pedestal Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  cout <<"MADE HEADER"<<endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  histoHTML2(runNo,ProblemPedestals,"i#Eta","i#Phi", 92, htmlFile,htmlDir);
  htmlFile<<"</tr>"<<endl;
  htmlFile<<"<tr><td> A pedestal is considered problematic if its mean differs from the nominal mean value of "<<nominalPedMeanInADC_<<" by more than "<<maxPedMeanDiffADC_<<" ADC counts.<br>";
  htmlFile<<"It is also considered problematic if its RMS differs from the nominal RMS of "<<nominalPedWidthInADC_<<" by more than "<<maxPedWidthDiffADC_<<" ADC counts.<br>"<<endl;
  htmlFile<<"<tr><td><a href=\"Expert_"<< htmlName<<"\">ExpertPlots </a></br></\td>"<<endl;
  htmlFile<<"</tr></table><br>"<<endl;
  
  cout <<"MADE MAIN PLOT"<<endl;

  // Now print out problem cells
  
  htmlFile <<"<td> Problem Cells</td><td align=\"center\"> Fraction of Events in which cells are bad (%)</td><td> Mean (ADC)</td><td>< RMS (ADC)</td></tr>"<<endl;

  cout <<"GETTING BINS"<<endl;
  if (ProblemPedestals==0)
    {
      cout <<"ERROR; don't have ProblemPEdestal!"<<endl;
      return;
    }
  int etabins  = ProblemPedestals->GetNbinsX();
  cout <<"ETA= "<<etabins<<endl;
  int phibins  = ProblemPedestals->GetNbinsY();
  cout <<"PHI= "<<phibins<<endl;
  float etaMin = ProblemPedestals->GetXaxis()->GetXmin();
  cout <<"ETAMIN = "<<etaMin<<endl;
  float phiMin = ProblemPedestals->GetYaxis()->GetXmin();
  cout <<"phimin = "<<phiMin<<endl;

  cout <<"TEST1"<<endl;
  int eta,phi;

  ostringstream name;
  cout <<"TEST 2"<<endl;
  for (int depth=0;depth<6; ++depth)
    {
      cout <<"DEPTH "<<depth<<endl;
      for (int ieta=1;ieta<=etabins;++ieta)
        {
          for (int iphi=1; iphi<=phibins;++iphi)
            {
              eta=ieta+int(etaMin)-1;
              phi=iphi+int(phiMin)-1;
	      if (ProblemPedestalsByDepth[depth]==0)
		{
		  cout <<"ERROR; couldn't get depth "<<depth<<endl;
		  continue;
		}
	      if (ProblemPedestalsByDepth[depth]->GetBinContent(ieta,iphi)>0)
		{
		  if (depth<2)
		    (fabs(eta)<29) ? name<<"HB" : name<<"HF";
		  else if (depth==3)
		    (fabs(eta)<42) ? name<<"HO" : name<<"ZDC";
		  else name <<"HE";
		  cout <<"NAME = "<<name.str().c_str()<<endl;
		  htmlFile<<"<td>"<<name.str().c_str()<<"</td><td>"<<ProblemPedestalsByDepth[depth]->GetBinContent(ieta,iphi)<<"</td><td> TBA </td>  <td></td></tr>"<<endl;
		  name.str("");
		}
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // for (int depth=0;...)
  
  
  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();
  
  return;
}



void HcalPedestalClient::loadHistograms(TFile* infile)
{
  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/PedestalMonitor/Pedestal Task Event Number");
  if(tnd)
    {
      string s =tnd->GetTitle();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    }

   ostringstream name;
  // Grab individual histograms
  name<<process_.c_str()<<"Hcal/PedestalMonitor/ ProblemPedestals";
  ProblemPedestals = (TH2F*)infile->Get(name.str().c_str());
  name.str("");
  
  for (int i=0;i<6;++i)
    {
      // Grab arrays of histograms
      name<<process_.c_str()<<"Hcal/PedestalMonitor/problemPedestals/"<<subdets_[i]<<"Problem Pedestal Rate";
      ProblemPedestalsByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");

      name<<process_.c_str()<<"Hcal/PedestalMonitor/"<<subdets_[i]<<"Pedestal Mean Map (ADC)";
      MeanMapByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
      
      name<<process_.c_str()<<"Hcal/PedestalMonitor/"<<subdets_[i]<<"Pedestal RMS Map (ADC)";
      RMSMapByDepth[i] = (TH2F*)infile->Get(name.str().c_str());
      name.str("");
    }
  return;
} // void HcalPedestalClient::loadHistograms(...)

