#include <DQM/HcalMonitorClient/interface/HcalRecHitClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalRecHitClient::HcalRecHitClient(){}

void HcalRecHitClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);
  ievt_ = 0;
  jevt_ = 0;
  for(int i=0; i<4; i++){
    occ_[i]=0; energy_[i]=0;
    energyT_[i]=0; time_[i]=0;
    tot_occ_[i]=0;
  }
  hfshort_E_all=0;
  //  hfshort_E_low=0;
  hfshort_T_all=0;

  tot_energy_=0;

}

HcalRecHitClient::~HcalRecHitClient(){
  this->cleanup();
}

void HcalRecHitClient::beginJob(void){

  if ( debug_ ) cout << "HcalRecHitClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  this->setup();
  this->resetAllME();
  return;
}

void HcalRecHitClient::beginRun(void){

  if ( debug_ ) cout << "HcalRecHitClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
}

void HcalRecHitClient::endJob(void) {

  if ( debug_ ) cout << "HcalRecHitClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup(); 
  return;
}

void HcalRecHitClient::endRun(void) {

  if ( debug_ ) cout << "HcalRecHitClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();
  return;
}

void HcalRecHitClient::setup(void) {
  return;
}

void HcalRecHitClient::cleanup(void) {
  
  if(cloneME_){
    for(int i=0; i<4; i++){
      if(occ_[i]) delete occ_[i];
      if(energy_[i]) delete energy_[i];
      if(energyT_[i]) delete energyT_[i];
      if(time_[i]) delete time_[i];
      if(tot_occ_[i]) delete tot_occ_[i];
    } 

    if(hfshort_E_all) delete hfshort_E_all;
    //if(hfshort_E_low) delete hfshort_E_low;
    if(hfshort_T_all) delete hfshort_T_all;
    
    if(tot_energy_) delete tot_energy_;
  }  
  
  for(int i=0; i<4; i++){
    occ_[i]=0; energy_[i]=0;
    energyT_[i]=0; time_[i]=0;
    tot_occ_[i]=0;
  }
  hfshort_E_all=0;
  //hfshort_E_low=0;
  hfshort_T_all=0;

  tot_energy_=0;
  

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  return;
}


void HcalRecHitClient::report(){
  if(!dbe_) return;
  if ( debug_ ) cout << "HcalRecHitClient: report" << endl;
  this->setup();

  char name[256];
  sprintf(name, "%sHcal/RecHitMonitor/RecHit Event Number",process_.c_str());
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

void HcalRecHitClient::analyze(void){

  jevt_++;
  int updates = 0;

  if ( updates % 10 == 0 ) {
    if ( debug_ ) cout << "HcalRecHitClient: " << updates << " updates" << endl;
  }

  return;
}

void HcalRecHitClient::getHistograms(){
  if(!dbe_) return;
  char name[150];    
  for(int i=0; i<4; i++){
    sprintf(name,"RecHitMonitor/RecHit Depth %d Occupancy Map",i+1);
    tot_occ_[i] = getHisto2(name, process_, dbe_, debug_,cloneME_);
  }

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 
    
    sprintf(name,"RecHitMonitor/%s/%s RecHit Total Energy",type.c_str(),type.c_str());      
    energyT_[i] = getHisto(name, process_,dbe_,debug_,cloneME_);

    if(i==2){
      sprintf(name,"RecHitMonitor/%s/%s Long RecHit Energies",type.c_str(),type.c_str());      
      energy_[i] = getHisto(name, process_,dbe_,debug_,cloneME_);
      
      sprintf(name,"RecHitMonitor/%s/%s Long RecHit Times",type.c_str(),type.c_str());      
      time_[i] = getHisto(name, process_,dbe_,debug_,cloneME_);
    } 
    else {
      sprintf(name,"RecHitMonitor/%s/%s RecHit Energies",type.c_str(),type.c_str());      
      energy_[i] = getHisto(name, process_,dbe_,debug_,cloneME_);
      
      sprintf(name,"RecHitMonitor/%s/%s RecHit Times",type.c_str(),type.c_str());      
      time_[i] = getHisto(name, process_,dbe_,debug_,cloneME_);
    }

    sprintf(name,"RecHitMonitor/%s/%s RecHit Geo Occupancy Map - Threshold",type.c_str(),type.c_str());
    occ_[i] = getHisto2(name, process_,dbe_,debug_,cloneME_);
  }


  sprintf(name,"RecHitMonitor/HF/HF Short RecHit Energies");
  hfshort_E_all = getHisto(name, process_,dbe_,debug_,cloneME_);

  //sprintf(name,"RecHitMonitor/HF/HF Short RecHit Energies - Low Region");
  //  hfshort_E_low = getHisto(name, process_,dbe_,debug_,cloneME_);

  sprintf(name,"RecHitMonitor/HF/HF Short RecHit Times");
  hfshort_T_all = getHisto(name, process_,dbe_,debug_,cloneME_);
  
  sprintf(name,"RecHitMonitor/RecHit Total Energy");   
  tot_energy_ = getHisto(name, process_,dbe_, debug_,cloneME_);

  return;
}

void HcalRecHitClient::resetAllME(){
  if(!dbe_) return;
  Char_t name[150];
  
  sprintf(name,"%sHcal/RecHitMonitor/RecHit Total Energy",process_.c_str());
  resetME(name,dbe_);
  for(int i=1; i<5; i++){
    sprintf(name,"%sHcal/RecHitMonitor/RecHit Depth %d Occupancy Map",process_.c_str(),i);
    resetME(name,dbe_);
    sprintf(name,"%sHcal/RecHitMonitor/RecHit Depth %d Energy Map",process_.c_str(),i);
    resetME(name,dbe_);
  }
  sprintf(name,"%sHcal/RecHitMonitor/RecHit Eta Occupancy Map",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/RecHitMonitor/RecHit Phi Occupancy Map",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/RecHitMonitor/RecHit Eta Energy Map",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/RecHitMonitor/RecHit Phi Energy Map",process_.c_str());
  resetME(name,dbe_);

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 

    sprintf(name,"%sHcal/RecHitMonitor/%s/%s RecHit Geo Occupancy Map - Threshold",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    resetME(name,dbe_);
    sprintf(name,"%sHcal/RecHitMonitor/%s/%s RecHit Total Energy",process_.c_str(),type.c_str(),type.c_str());   
 
    if(i==2){
      sprintf(name,"%sHcal/RecHitMonitor/%s/%s Long RecHit Energies",process_.c_str(),type.c_str(),type.c_str());      
      resetME(name,dbe_);
      sprintf(name,"%sHcal/RecHitMonitor/%s/%s Long RecHit Energies - Low Region",process_.c_str(),type.c_str(),type.c_str());
      resetME(name,dbe_);
      sprintf(name,"%sHcal/RecHitMonitor/%s/%s Long RecHit Times",process_.c_str(),type.c_str(),type.c_str()); 
      resetME(name,dbe_);
    }    
    else {
      sprintf(name,"%sHcal/RecHitMonitor/%s/%s RecHit Energies",process_.c_str(),type.c_str(),type.c_str());      
      resetME(name,dbe_);
      sprintf(name,"%sHcal/RecHitMonitor/%s/%s RecHit Energies - Low Region",process_.c_str(),type.c_str(),type.c_str());  
      resetME(name,dbe_);
      sprintf(name,"%sHcal/RecHitMonitor/%s/%s RecHit Times",process_.c_str(),type.c_str(),type.c_str()); 
      resetME(name,dbe_);
    }     
  }

  sprintf(name,"%sHcal/RecHitMonitor/HF/HF Short RecHit Energies",process_.c_str());
  resetME(name,dbe_);
  //sprintf(name,"%sHcal/RecHitMonitor/HF/HF Short RecHit Energies - Low Region",process_.c_str());
  //resetME(name,dbe_);
  sprintf(name,"%sHcal/RecHitMonitor/HF/HF Short RecHit Times",process_.c_str());
  resetME(name,dbe_);

  return;
}


void HcalRecHitClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  cout << "Preparing HcalRecHitClient html output ..." << endl;
  string client = "RecHitMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal RecHit Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal RecHits</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table  width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"RecHitMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"RecHitMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"RecHitMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal RecHit Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  if(subDetsOn_[0]) htmlFile << "<a href=\"#HB_Plots\">HB Plots </a></br>" << endl;
  if(subDetsOn_[1]) htmlFile << "<a href=\"#HE_Plots\">HE Plots </a></br>" << endl;
  if(subDetsOn_[2]) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  if(subDetsOn_[3]) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,tot_occ_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,tot_occ_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,tot_occ_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,tot_occ_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  //removed total energy for cosmics run
  //  htmlFile << "<tr align=\"left\">" << endl;
  //  histoHTML(runNo,tot_energy_,"Total Energy (GeV)","Events", 100, htmlFile,htmlDir);
  //  htmlFile << "</tr>" << endl;


  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 
    
    htmlFile << "<tr align=\"left\">" << endl;

    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML2(runNo,occ_[i],"iEta","iPhi", 92, htmlFile,htmlDir);
    //removed total energy for cosmics run
    //    histoHTML(runNo,energyT_[i],"Total Energy (GeV)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    if(i==2){
      htmlFile << "<tr align=\"left\">" << endl;
      histoHTML(runNo,energy_[i],"Long fibers, RecHit Energy (GeV)","Events", 92, htmlFile,htmlDir);
      histoHTML(runNo,time_[i],"Long fibers, RecHit Time (nS)","Events", 100, htmlFile,htmlDir);

      htmlFile << "<tr align=\"left\">" << endl;
      histoHTML(runNo,hfshort_E_all,"Short fibers, RecHit Energy (GeV)","Events", 92, htmlFile,htmlDir);
      histoHTML(runNo,hfshort_T_all,"Short fibers, RecHit Time (nS)","Events", 100, htmlFile,htmlDir);
    }
    else {
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(runNo,energy_[i],"RecHit Energy (GeV)","Events", 92, htmlFile,htmlDir);
    histoHTML(runNo,time_[i],"RecHit Time (nS)","Events", 100, htmlFile,htmlDir);
    }

    htmlFile << "</tr>" << endl;	
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;


  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  return;
}

void HcalRecHitClient::createTests(){
  if(!dbe_) return;

  if(debug_) printf("Creating RecHit tests...\n"); 
  
  return;
}

void HcalRecHitClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/RecHitMonitor/RecHit Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  char name[150];    
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 
    
    sprintf(name,"DQMData/Hcal/RecHitMonitor/%s/%s RecHit Total Energy",type.c_str(),type.c_str());      
    energyT_[i] = (TH1F*)infile->Get(name);

    if(i==2){
      sprintf(name,"DQMData/Hcal/RecHitMonitor/%s/%s Long RecHit Energies",type.c_str(),type.c_str());      
      energy_[i] = (TH1F*)infile->Get(name);
    
      sprintf(name,"DQMData/Hcal/RecHitMonitor/%s/%s Long RecHit Times",type.c_str(),type.c_str());      
      time_[i] = (TH1F*)infile->Get(name);
    } 
    else {
      sprintf(name,"DQMData/Hcal/RecHitMonitor/%s/%s RecHit Energies",type.c_str(),type.c_str());      
      energy_[i] = (TH1F*)infile->Get(name);
    
      sprintf(name,"DQMData/Hcal/RecHitMonitor/%s/%s RecHit Times",type.c_str(),type.c_str());      
      time_[i] = (TH1F*)infile->Get(name);
    }

    sprintf(name,"DQMData/Hcal/RecHitMonitor/%s/%s RecHit Geo Occupancy Map - Threshold",type.c_str(),type.c_str());
    occ_[i] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/RecHitMonitor/RecHit Depth %d Occupancy Map",i);
    tot_occ_[i] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/RecHitMonitor/%s/%s RecHit Times",type.c_str(),type.c_str());      
    time_[i] = (TH1F*)infile->Get(name);
    
  }

    //-3 extra histos for HF short:
    sprintf(name,"DQMData/Hcal/RecHitMonitor/HF/HF Short RecHit Energies");   
    hfshort_E_all= (TH1F*)infile->Get(name);
    //-not using this one for now: 
    //    sprintf(name,"DQMData/Hcal/RecHitMonitor/HF/HF Short RecHit Energies - Low Region");   
    //    hfshort_E_low= (TH1F*)infile->Get(name);
    sprintf(name,"DQMData/Hcal/RecHitMonitor/HF/HF Short RecHit Times");      
    hfshort_T_all = (TH1F*)infile->Get(name);


  sprintf(name,"DQMData/Hcal/RecHitMonitor/RecHit Total Energy");   
  tot_energy_ = (TH1F*)infile->Get(name);

  return;
}
