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
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 42.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -42.5);

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73.5);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", -0.5);

  checkHB_= ps.getUntrackedParameter<bool>("checkHB",true);
  checkHE_= ps.getUntrackedParameter<bool>("checkHE",true);
  checkHO_= ps.getUntrackedParameter<bool>("checkHO",true);
  checkHF_= ps.getUntrackedParameter<bool>("checkHF",true);


  phiBins_=(int)(abs(phiMax_-phiMin_));
  etaBins_=(int)(abs(etaMax_-etaMin_));

  // Summary maps
  meGlobalSummary_=0;

  // All initial status floats set to -1 (uncertain)
  // For the moment, these are just local variables; if we want to keep
  // them in the root file, we need to book them as MonitorElements
  status_HB_=-1;
  status_HE_=-1;
  status_HO_=-1;
  status_HF_=-1;
  status_global_=-1;
  
  subdetCells_.insert(make_pair("HB",2592));
  subdetCells_.insert(make_pair("HE",2592));
  subdetCells_.insert(make_pair("HO",2160));
  subdetCells_.insert(make_pair("HF",1728));
  
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
  lastupdate_=0; // keeps analyze from being called by both endRun and endJob
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
  // When the job ends, we want to make a summary before exiting
  if (ievt_>lastupdate_)
    analyze();
  this->cleanup();
} // void HcalSummaryClient::endJob(void)


void HcalSummaryClient::endRun(void) 
{
  if ( debug_ ) cout << "HcalSummaryClient: endRun, jevt = " << jevt_ << endl;
  // When the run ends, we want to make a summary before exiting
  analyze();
  lastupdate_=ievt_;
  this->cleanup();
} // void HcalSummaryClient::endRun(void) 


void HcalSummaryClient::setup(void)
{

  char histo[200];

  // Is this the correct folder?
  dqmStore_->setCurrentFolder( prefixME_ + "/HcalSummaryClient" );


  // This histogram may be redundant?
  sprintf(histo,"Global Summary");
  meGlobalSummary_ = dqmStore_->book2D(histo, histo, etaBins_,etaMin_,etaMax_,
					 phiBins_,phiMin_,phiMax_);
  meGlobalSummary_->setAxisTitle("i#eta", 1);
  meGlobalSummary_->setAxisTitle("i#phi", 2);


  
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

  sprintf(histo,"HBstatus");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->bookFloat(histo);

  sprintf(histo,"HEstatus");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->bookFloat(histo);

  sprintf(histo,"HOstatus");
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/" + histo) )
    {
      dqmStore_->removeElement(me->getName());
    }
  me = dqmStore_->bookFloat(histo);

  sprintf(histo,"HFstatus");
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
  me = dqmStore_->book2D(histo, histo, etaBins_,etaMin_,etaMax_,
			 phiBins_,phiMin_,phiMax_);
  
  me->setAxisTitle("i#eta", 1);
  me->setAxisTitle("i#phi", 2);
 
} // void HcalSummaryClient::setup(void)


void HcalSummaryClient::cleanup(void) 
{
  
  if ( ! enableCleanup_ ) return;

  MonitorElement* me;

  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary") ) {
    dqmStore_->removeElement(me->getName());
  }
  

  /*
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap") ) {
    dqmStore_->removeElement(me->getName());
  }
  */  

  // redundant?  Handled above?
  if (meGlobalSummary_) dqmStore_->removeElement(meGlobalSummary_->getName());
  meGlobalSummary_=0;
  

} // void HcalSummaryClient::cleanup(void) 



void HcalSummaryClient::incrementCounters(void)
{
  ievt_++;
  jevt_++;
  return;
}

void HcalSummaryClient::analyze(void)
{

  if ( ievt_ % 10 == 0 ) 
    {
      //if ( debug_ )
      cout << "HcalSummaryClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
    }

  if (checkHB_) analyze_deadcell("HB",status_HB_);
  if (checkHE_) analyze_deadcell("HE",status_HE_);
  if (checkHO_) analyze_deadcell("HO",status_HO_);
  if (checkHF_) analyze_deadcell("HF",status_HF_);



  MonitorElement* me;
  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo" );

  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummary");
  if (me) me->Fill(status_global_);

  dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/HBstatus") )
    me->Fill(status_HB_);
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/HEstatus") )
    me->Fill(status_HE_);
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/HOstatus") )
    me->Fill(status_HO_);
  if ( me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/HFstatus") )
    me->Fill(status_HF_);
  
  dqmStore_->setCurrentFolder( prefixME_);

  return;
} //void HcalSummaryClient::analyze(void)



float HcalSummaryClient::analyze_deadcell(std::string subdetname, float& subdet)
{
  float status = -1;
  //dqmStore_->setCurrentFolder( prefixME_ + "/EventInfo/reportSummaryContents" );
  MonitorElement* me;
  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  if (!me)
    return status;

  char name[150];
  sprintf(name,"%s/DeadCellMonitor/%s/%sProblemDeadCells",prefixME_.c_str(),
	  subdetname.c_str(),subdetname.c_str());
   MonitorElement* me_temp = dqmStore_->get(name); // get Monitor Element named 'name'
  if (!me_temp) return status;

  //TH2F* temp = (TH2F*)me_temp->getTH2F();
  //float maxval=temp->GetMaximum(); // scale factor for histogram

  double origbincontent=0; // stores value from report
  double newbincontent=0;
  float badcells=0.; 
  float eta, phi;

  for (int ieta=1;ieta<=etaBins_;++ieta)
    {   
      eta=ieta+int(etaMin_)-1;
      if (eta==0) continue; // skip eta=0 bin -- unphysical
      if (abs(eta)>41) continue; // skip unphysical "boundary" bins in histogram

      for (int iphi=1; iphi<=phiBins_;++iphi)
	{
	  origbincontent=me->getBinContent(ieta,iphi);
	  //newbincontent=temp->GetBinContent(ieta,iphi)/maxval;
	  newbincontent=me_temp->getBinContent(ieta,iphi)/ievt_; // normalize to number of events

	  phi=iphi+int(phiMin_)-1;
	  
	  //newbincontent=me_temp->getBinContent(ieta,iphi)/maxval;
	  if (origbincontent==-1)
	    me->setBinContent(ieta,iphi,newbincontent);
	  else
	    //me->setBinContent(ieta,iphi,min(1., newbincontent));
	    if (newbincontent>0)
	      {
		//me->Fill(eta,phi,newbincontent);
		me->setBinContent(ieta,iphi,min(1.,newbincontent));
	      }

	  if (newbincontent>0)
	    {
	      badcells+=newbincontent;
	    }
	} // loop over iphi
    } // loop over ieta
  
  // Normalize badcells to give avg # of bad cells per event
  badcells=1.*badcells/ievt_;
  
  std::map<std::string, int>::const_iterator it;
  it =subdetCells_.find(subdetname);
  // didn't find subdet in map
  if (it==subdetCells_.end())
    return -1;
  if (it->second == 0 || (it->second)<badcells)
    return -1;

  // Status is 1 if no bad cells found
  // Otherwise, status = 1 - (avg fraction of bad cells/event)
  status=1.-(1.*badcells)/it->second;
 
 // The only way to change the overall subdet, global status words is 
  // if the deadcell status word is reasonable (i.e., not = -1)
  if (subdet==-1)
    subdet=status;
  else
    subdet*=status;
  if (status_global_==-1)
    status_global_=status;
  else
    status_global_*=status;

  return status;
} // void HcalSummaryClient::analyze_deadcell(std::string subdetname)


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
