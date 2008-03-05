#include <DQM/HcalMonitorClient/interface/HcalCaloTowerClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalCaloTowerClient::HcalCaloTowerClient(){}

void HcalCaloTowerClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);
  ievt_ = 0;
  jevt_ = 0;
  occ_=0;
  energy_=0;

}

HcalCaloTowerClient::~HcalCaloTowerClient(){
  this->cleanup();
}

void HcalCaloTowerClient::beginJob(void){

  if ( debug_ ) cout << "HcalCaloTowerClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  this->setup();
  this->resetAllME();
  return;
}

void HcalCaloTowerClient::beginRun(void){

  if ( debug_ ) cout << "HcalCaloTowerClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
}

void HcalCaloTowerClient::endJob(void) {

  if ( debug_ ) cout << "HcalCaloTowerClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup(); 
  return;
}

void HcalCaloTowerClient::endRun(void) {

  if ( debug_ ) cout << "HcalCaloTowerClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();
  return;
}

void HcalCaloTowerClient::setup(void) {
  return;
}

void HcalCaloTowerClient::cleanup(void) {
  
  if(cloneME_)
    {
      if(occ_) delete occ_;
      if(energy_) delete energy_;
    }    

  
  occ_=0; 
  energy_=0;
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  return;
}


void HcalCaloTowerClient::report(){
  if(!dbe_) return;
  if ( debug_ ) cout << "HcalCaloTowerClient: report" << endl;
  this->setup();

  char name[256];
  sprintf(name, "%sHcal/CaloTowerMonitor/CaloTower Event Number",process_.c_str());
  MonitorElement* me = dbe_->get(name); 
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_ ) cout << "Found '" << name << "'" << endl;
  }
  getHistograms();

  return;
}

void HcalCaloTowerClient::analyze(void){

  jevt_++;
  int updates = 0;

  if ( updates % 10 == 0 ) {
    if ( debug_ ) cout << "HcalCaloTowerClient: " << updates << " updates" << endl;
  }

  return;
}

void HcalCaloTowerClient::getHistograms(){
  if(!dbe_) return;
  char name[150];    
  sprintf(name,"CaloTowerMonitor/CaloTowerOccupancy");
  occ_=getHisto2(name,process_,dbe_, debug_,cloneME_);
  sprintf(name,"CaloTowerMonitor/CaloTowerEnergy");
  energy_=getHisto2(name,process_,dbe_, debug_,cloneME_);

  return;
}

void HcalCaloTowerClient::resetAllME(){
  if(!dbe_) return;
  Char_t name[150];
  
  sprintf(name,"%sHcal/CaloTowerMonitor/CaloTowerOccupancy",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/CaloTowerMonitor/CaloTowerEnergyy",process_.c_str());
  resetME(name,dbe_);


  return;
}


void HcalCaloTowerClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  cout << "Preparing HcalCaloTowerClient html output ..." << endl;
  string client = "CaloTowerMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal CaloTower Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal CaloTowers</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table  width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"CaloTowerMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"CaloTowerMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"CaloTowerMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal CaloTower Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  /*
  if(subDetsOn_[0]) htmlFile << "<a href=\"#HB_Plots\">HB Plots </a></br>" << endl;
  if(subDetsOn_[1]) htmlFile << "<a href=\"#HE_Plots\">HE Plots </a></br>" << endl;
  if(subDetsOn_[2]) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  if(subDetsOn_[3]) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  */
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,occ_,"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,energy_,"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile<<"</table>"<<endl;
  htmlFile << "<br>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  return;
}


void HcalCaloTowerClient::createTests(){
  if(!dbe_) return;

  if(debug_) printf("Creating CaloTower tests...\n"); 
  
  return;
}


void HcalCaloTowerClient::loadHistograms(TFile* infile){

  // Don't currently use event number
  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/CaloTowerMonitor/CaloTower Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  char name[150];    
  sprintf(name,"DQMData/Hcal/CaloTowerMonitor/CaloTowerOccupancy");      
  occ_= (TH2F*)infile->Get(name);
  sprintf(name,"DQMData/Hcal/CaloTowerMonitor/CaloTowerEnergy");      
  energy_= (TH2F*)infile->Get(name);

  return;
}
