#include <memory>
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>

#include "TCanvas.h"
#include "TStyle.h"

#include "DQMServices/Core/interface/DQMStore.h"


#include "DQM/HcalMonitorClient/interface/HcalMonitorClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDataFormatClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDigiClient.h"
#include "DQM/HcalMonitorClient/interface/HcalRecHitClient.h"
#include "DQM/HcalMonitorClient/interface/HcalTrigPrimClient.h"
#include "DQM/HcalMonitorClient/interface/HcalPedestalClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDeadCellClient.h"
#include "DQM/HcalMonitorClient/interface/HcalHotCellClient.h"

#include "DQM/HcalMonitorClient/interface/HcalSummaryClient.h"


using namespace cms;
using namespace edm;
using namespace std;


HcalSummaryClient::HcalSummaryClient(const ParameterSet& ps)
{
  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "Hcal");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // eta, phi boundaries -- need to put them in client as well as monitor?
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 41.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -41.5);

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73.5);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", -0.5);

  phiBins_=(int)(abs(phiMax_-phiMin_));
  etaBins_=(int)(abs(etaMax_-etaMin_));


  // Summary maps
  meGlobalSummary_=0;
  meHotCellMap_=0;
  meDeadCellMap_=0;

} // HcalSummaryClient::HcalSummaryClient(const ParameterSet& ps)


HcalSummaryClient::~HcalSummaryClient()
{
}

void HcalSummaryClient::beginJob(DQMStore* dqmStore)
{
  dqmStore_=dqmStore;
  //  if (debug_) 
  cout <<"HcalSummaryClient: beginJob"<<endl;
  ievt_ = 0; // keepts track of all events in job
  jevt_ = 0; // keeps track of all events in run

} // void HcalSummaryClient::beginJob(DQMStore* dqmStore)


void HcalSummaryClient::beginRun(void)
{
  if ( debug_ ) cout << "HcalSummaryClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();
} //void HcalSummaryClient::beginRun(void)


void HcalSummaryClient::endJob(void)
{
  if ( debug_ ) cout << "HcalSummaryClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();
} // void HcalSummaryClient::endJob(void)


void HcalSummaryClient::endRun(void) 
{
  if ( debug_ ) cout << "HcalSummaryClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();
} // void HcalSummaryClient::endRun(void) 


void HcalSummaryClient::setup(void)
{

  char histo[200];

  // Is this the correct folder?
  dqmStore_->setCurrentFolder( prefixME_ + "/HcalSummaryClient" );

  // remove old versions of summary plots
  if (meHotCellMap_) dqmStore_->removeElement( meHotCellMap_->getName());
  if (meDeadCellMap_) dqmStore_->removeElement( meDeadCellMap_->getName());


  sprintf(histo, "Hcal Hot Cells");
  meHotCellMap_ = dqmStore_->book2D(histo, histo, etaBins_,etaMin_,etaMax_,
				phiBins_,phiMin_,phiMax_);
  meHotCellMap_->setAxisTitle("ieta", 1);
  meHotCellMap_->setAxisTitle("iphi", 2);
    
  sprintf(histo, "Hcal Dead Cells");
  meDeadCellMap_ = dqmStore_->book2D(histo, histo, etaBins_,etaMin_,etaMax_,
				 phiBins_,phiMin_,phiMax_);
  meDeadCellMap_->setAxisTitle("ieta", 1);
  meDeadCellMap_->setAxisTitle("iphi", 2);

  // This histogram may be redundant?
  sprintf(histo,"Global Summary");
  meGlobalSummary_ = dqmStore_->book2D(histo, histo, etaBins_,etaMin_,etaMax_,
					 phiBins_,phiMin_,phiMax_);
  meGlobalSummary_->setAxisTitle("ieta", 1);
  meGlobalSummary_->setAxisTitle("iphi", 2);


  
  // Monitor Elements in required format, according to https://twiki.cern.ch/twiki/bin/view/CMS/SummaryDisplayProposal
  MonitorElement* me;

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  sprintf(histo, "reportSummary");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/" + histo) ) {
    dqmStore_->removeElement(me->getName());
  }
  me = dqmStore_->bookFloat(histo);

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );


  // Create floats showing subtasks status
  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );
  sprintf(histo,"hot cell status");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->bookFloat(histo);

  sprintf(histo,"dead cell status");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->bookFloat(histo);



  // Create global summary map
  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  sprintf(histo, "reportSummaryMap");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/" + histo) ) 
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->book2D(histo, histo, 72, 0., 72., 34, 0., 34);

  me->setAxisTitle("ieta", 1);
  me->setAxisTitle("iphi", 2);
 

 

} // void HcalSummaryClient::setup(void)


void HcalSummaryClient::cleanup(void) 
{
  
  if ( ! enableCleanup_ ) return;

  MonitorElement* me;

  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary") ) {
    dqmStore_->removeElement(me->getName());
  }
  

  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap") ) {
    dqmStore_->removeElement(me->getName());
  }
  
  // redundant?  Handled above?
  if (meGlobalSummary_) dqmStore_->removeElement(meGlobalSummary_->getName());
  meGlobalSummary_=0;
  

  // Get rid of hot/dead cell maps
  if (meHotCellMap_) dqmStore_->removeElement(meHotCellMap_->getName());
  meHotCellMap_=0;
  
  if (meDeadCellMap_) dqmStore_->removeElement(meDeadCellMap_->getName());
  meDeadCellMap_=0;

} // void HcalSummaryClient::cleanup(void) 



void HcalSummaryClient::analyze(void)
{
  
  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) 
    {
      //if ( debug_ )
      cout << "HcalSummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
    }
  for (int ieta=1;ieta<=etaBins_;++ieta)
    {
      for (int iphi=1; iphi<=phiBins_;++iphi)
	{
	  // Set all hot/dead cell values to -1 to start
	  // In the future, we'll need to get the Maps from ... task? client? and read their entry values here
	  meHotCellMap_->setBinContent(ieta,iphi,-1);
	  meDeadCellMap_->setBinContent(ieta,iphi,-1);
	} // loop over iphi
    } // loop over ieta


  // Set report summary stuff -- everything set to "unknown" (-1) for the moment

  MonitorElement* me;

  float reportSummary = -1.0;
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
  if (me) me->Fill(reportSummary);

  char histo[200];


  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );
  sprintf(histo,"hot cell status");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    me->Fill(-1);
  sprintf(histo,"dead cell status");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    me->Fill(-1);

  
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  if (me)
    {
      for (int ieta=1;ieta<=etaBins_;++ieta)
	{
	  for (int iphi=1; iphi<=phiBins_;++iphi)
	    {
	      // Doesn't do anything real yet
	      me->setBinContent(ieta,iphi,-1);
	    } // loop over iphi
	} // loop over ieta
    } // if (me)


} //void HcalSummaryClient::analyze(void)





void HcalSummaryClient::htmlOutput(int run, string& htmlDir, string& htmlName)
{

  if ( verbose_ ) cout << "Preparing HcalSummaryClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:Summary output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  //htmlFile << "<br>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">SUMMARY</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<br>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  // values taken from EBSummaryClient.cc
  const int csize = 400;

  TCanvas* cMap = new TCanvas("cMap", "Temp", int(360./170.*csize), csize);
  //TCanvas* cMapPN = new TCanvas("cMapPN", "Temp", int(360./170.*csize), int(20./90.*360./170.*csize));

  //  const double histMax = 1.e15;

  TH2F* obj2f;
  std::string imgNameMap="";
  std::string imgName;
  gStyle->SetPaintTextFormat("+g");

  std::string meName;


  // Test for now -- let's just dump out global summary histogram
  obj2f =  meGlobalSummary_->getTH2F();
  if (obj2f && obj2f->GetEntries()!=0)
    {

      meName = obj2f->GetName();

      replace(meName.begin(), meName.end(), ' ', '_');
      imgNameMap = meName + ".png";
      imgName = htmlDir + imgNameMap;
      cMap->cd();
      gStyle->SetOptStat(" ");
      gStyle->SetPalette(1);
      //gStyle->SetPalette(6, pCol3);
      cMap->SetGridx();
      cMap->SetGridy();

      obj2f->GetXaxis()->SetLabelSize(0.03);
      obj2f->GetYaxis()->SetLabelSize(0.03);
      
      cMap->Update();
      cMap->SaveAs(imgName.c_str());

    } // if (obj2f && obj2f->GetEntries()!=0)

  gStyle->SetPaintTextFormat();
  
  if ( imgNameMap.size() != 0 ) 
    {
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
      htmlFile << "<tr align=\"center\">" << endl;
      htmlFile << "<td><img src=\"" << imgNameMap << "\" usemap=\"#Integrity\" border=0></td>" << endl;
      htmlFile << "</tr>" << endl;
      htmlFile << "</table>" << endl;
      htmlFile << "<br>" << endl;
    } // if ( imgNameMap.size() != 0 ) 

  htmlFile.close();

} // void HcalSummaryClient::htmlOutput(int run, string& htmlDir, string& htmlName)
