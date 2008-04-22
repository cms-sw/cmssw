#include <DQM/HcalMonitorClient/interface/HcalDataFormatClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalDataFormatClient::HcalDataFormatClient(){}


void HcalDataFormatClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  for(int i=0; i<3; i++){
    dferr_[i] = NULL;
  }
  
  spigotErrs_ = NULL;
  DCC_Err_Warn_ = NULL;
  DCC_Evt_Fmt_ = NULL;
  CDF_Violation_ = NULL;
  DCC_Spigot_Err_ = NULL;
  badDigis_ = NULL;
  unmappedDigis_ = NULL;
  unmappedTPDs_ = NULL;
  fedErrMap_ = NULL;
  BCN_ = NULL;
  dccBCN_ = NULL;
  BCNCheck_ = NULL;
  EvtNCheck_ = NULL;
  FibOrbMsgBCN_ = NULL;
  
  BCNMap_ = NULL;
  EvtMap_ = NULL;
  ErrMapbyCrate_ = NULL;
  FWVerbyCrate_ = NULL;
  ErrCrate0_ = NULL;
  ErrCrate1_ = NULL;
  ErrCrate2_ = NULL;
  ErrCrate3_ = NULL;
  ErrCrate4_ = NULL;
  ErrCrate5_ = NULL;
  ErrCrate6_ = NULL;
  ErrCrate7_ = NULL;
  ErrCrate8_ = NULL;
  ErrCrate9_ = NULL;
  ErrCrate10_ = NULL;
  ErrCrate11_ = NULL;
  ErrCrate12_ = NULL;
  ErrCrate13_ = NULL;
  ErrCrate14_ = NULL;
  ErrCrate15_ = NULL;
  ErrCrate16_ = NULL;
  ErrCrate17_ = NULL;
  InvHTRData_ = NULL;
  
}

HcalDataFormatClient::~HcalDataFormatClient(){
  this->cleanup();  
}

void HcalDataFormatClient::beginJob(void){
  if ( debug_ ) cout << "HcalDataFormatClient: beginJob" << endl;

  ievt_ = 0; jevt_ = 0;
  return;
}

void HcalDataFormatClient::beginRun(void){
  if ( debug_ ) cout << "HcalDataFormatClient: beginRun" << endl;

  jevt_ = 0;
  this->resetAllME();
  return;
}

void HcalDataFormatClient::endJob(void) {
  if ( debug_ ) cout << "HcalDataFormatClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

  return;
}

void HcalDataFormatClient::endRun(void) {

  if ( debug_ ) cout << "HcalDataFormatClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

  return;
}



void HcalDataFormatClient::cleanup(void) {

  if ( cloneME_ ) {
    for(int i=0; i<3; i++){
      if ( dferr_[i] ) delete dferr_[i];
    }
  
    if ( spigotErrs_) delete spigotErrs_;
    if ( DCC_Err_Warn_) delete DCC_Err_Warn_;
    if ( DCC_Evt_Fmt_) delete DCC_Evt_Fmt_;
    if ( DCC_Spigot_Err_) delete DCC_Spigot_Err_;
    if ( CDF_Violation_) delete CDF_Violation_;
    if ( badDigis_) delete badDigis_;
    if ( unmappedDigis_) delete unmappedDigis_;
    if ( unmappedTPDs_) delete unmappedTPDs_;
    if ( fedErrMap_) delete fedErrMap_;

    if( BCN_) delete BCN_;
    if( dccBCN_) delete dccBCN_;
    if( BCNCheck_) delete BCNCheck_;
    if( EvtNCheck_) delete EvtNCheck_;
    if( FibOrbMsgBCN_) delete FibOrbMsgBCN_;

   if (BCNMap_) delete BCNMap_;
   if (EvtMap_) delete EvtMap_;
   if (ErrMapbyCrate_) delete ErrMapbyCrate_;
   if (FWVerbyCrate_) delete FWVerbyCrate_;
   if (ErrCrate0_) delete ErrCrate0_;
   if (ErrCrate1_) delete ErrCrate1_;
   if (ErrCrate2_) delete ErrCrate2_;
   if (ErrCrate3_) delete ErrCrate3_;
   if (ErrCrate4_) delete ErrCrate4_;
   if (ErrCrate5_) delete ErrCrate5_;
   if (ErrCrate6_) delete ErrCrate6_;
   if (ErrCrate7_) delete ErrCrate7_;
   if (ErrCrate8_) delete ErrCrate8_;
   if (ErrCrate9_) delete ErrCrate9_;
   if (ErrCrate10_) delete ErrCrate10_;
   if (ErrCrate11_) delete ErrCrate11_;
   if (ErrCrate12_) delete ErrCrate12_;
   if (ErrCrate13_) delete ErrCrate13_;
   if (ErrCrate14_) delete ErrCrate14_;
   if (ErrCrate15_) delete ErrCrate15_;
   if (ErrCrate16_) delete ErrCrate16_;
   if (ErrCrate17_) delete ErrCrate17_;
   if (InvHTRData_) delete InvHTRData_;

  }  
  for(int i=0; i<3; i++){
    dferr_[i] = NULL;
  }
  
  spigotErrs_ = NULL;
  DCC_Err_Warn_ = NULL;  
  DCC_Evt_Fmt_ = NULL;
  CDF_Violation_ = NULL;
  DCC_Spigot_Err_ = NULL;
  badDigis_ = NULL;
  unmappedDigis_ = NULL;
  unmappedTPDs_ = NULL;
  fedErrMap_ = NULL;


  BCN_ = NULL;
  dccBCN_ = NULL;

  BCNCheck_ = NULL;
  EvtNCheck_ = NULL;
  FibOrbMsgBCN_ = NULL; 

   BCNMap_ = NULL;
   EvtMap_ = NULL;
   ErrMapbyCrate_ = NULL;
   FWVerbyCrate_ = NULL;
   ErrCrate0_ = NULL;
   ErrCrate1_ = NULL;
   ErrCrate2_ = NULL;
   ErrCrate3_ = NULL;
   ErrCrate4_ = NULL;
   ErrCrate5_ = NULL;
   ErrCrate6_ = NULL;
   ErrCrate7_ = NULL;
   ErrCrate8_ = NULL;
   ErrCrate9_ = NULL;
   ErrCrate10_ = NULL;
   ErrCrate11_ = NULL;
   ErrCrate12_ = NULL;
   ErrCrate13_ = NULL;
   ErrCrate14_ = NULL;
   ErrCrate15_ = NULL;
   ErrCrate16_ = NULL;
   ErrCrate17_ = NULL;
   InvHTRData_ = NULL;

  return;
}



void HcalDataFormatClient::analyze(void){
  jevt_++;

  int updates = 0;
  if ( updates % 10 == 0 ) {
    if ( debug_ ) cout << "HcalDataFormatClient: " << updates << " updates" << endl;
  }

  return;
}

void HcalDataFormatClient::getHistograms(){

  if(!dbe_) return;
  
  char name[150];     
  sprintf(name,"DataFormatMonitor/DCC Error and Warning");
  DCC_Err_Warn_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/DCC Event Format violation");
  DCC_Evt_Fmt_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/DCC Nonzero Spigot Conditions");
  DCC_Spigot_Err_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/Common Data Format violations");
  CDF_Violation_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/Spigot Format Errors");
  spigotErrs_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/Num Bad Quality Digis -DV bit-Err bit-Cap Rotation");
  badDigis_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/Num Unmapped Digis");
  unmappedDigis_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/Num Unmapped Trigger Primitive Digis");
  unmappedTPDs_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/FED Error Map from Unpacker Report");
  fedErrMap_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/BCN from HTRs");
  BCN_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/BCN from DCCs");
  dccBCN_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/BCN Difference Between Ref HTR and DCC");
  BCNCheck_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/EvN Difference Between Ref HTR and DCC");
  EvtNCheck_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/BCN of Fiber Orbit Message");
  FibOrbMsgBCN_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/EvN Inconsistent - HTR vs Ref HTR");
  EvtMap_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/BCN Inconsistent - HTR vs Ref HTR");
  BCNMap_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/HTR Firmware Version");
  FWVerbyCrate_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/Invalid HTR Data");
  InvHTRData_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/HTR Error Word by Crate");
  ErrMapbyCrate_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrMapbyCrate_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 0");
  ErrCrate0_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate0_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 1");
  ErrCrate1_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate1_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 2");
  ErrCrate2_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate2_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 3");
  ErrCrate3_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate3_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 4");
  ErrCrate4_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate4_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 5");
  ErrCrate5_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate5_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 6");
  ErrCrate6_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate6_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 7");
  ErrCrate7_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate7_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 8");
  ErrCrate8_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate8_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 9");
  ErrCrate9_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate9_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 10");
  ErrCrate10_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate10_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 11");
  ErrCrate11_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate11_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 12");
  ErrCrate12_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate12_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 13");
  ErrCrate13_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate13_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 14");
  ErrCrate14_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate14_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 15");
  ErrCrate15_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate15_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 16");
  ErrCrate16_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate16_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 17");
  ErrCrate17_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
  labelyBits(ErrCrate17_);
 
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HBHE";
    if(i==1) type = "HBHE";
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";
    sprintf(name,"DataFormatMonitor/%s Data Format Error Word", type.c_str());
    int ind = i-1;
    if (ind <0) ind = 0;
    dferr_[ind] = getHisto(name, process_, dbe_, debug_,cloneME_);    
    labelxBits(dferr_[ind]);
  }
  return;
}


void HcalDataFormatClient::labelxBits(TH1F* hist){
  
  if(hist==NULL) return;

  //hist->LabelsOption("v","X");

  hist->SetXTitle("Error Bit");
  hist->GetXaxis()->SetBinLabel(1,"Overflow Warn");
  hist->GetXaxis()->SetBinLabel(2,"Buffer Busy");
  hist->GetXaxis()->SetBinLabel(3,"Empty Event");
  hist->GetXaxis()->SetBinLabel(4,"Reject L1A");
  hist->GetXaxis()->SetBinLabel(5,"Latency Err");
  hist->GetXaxis()->SetBinLabel(6,"Latency Warn");
  hist->GetXaxis()->SetBinLabel(7,"OpDat Err");
  hist->GetXaxis()->SetBinLabel(8,"Clock Err");
  hist->GetXaxis()->SetBinLabel(9,"Bunch Err");
  hist->GetXaxis()->SetBinLabel(13,"Test Mode");
  hist->GetXaxis()->SetBinLabel(14,"Histo Mode");
  hist->GetXaxis()->SetBinLabel(15,"Calib Trig");
  hist->GetXaxis()->SetBinLabel(16,"Bit15 Err");
  
  return;
}

void HcalDataFormatClient::labelyBits(TH2F* hist){
  
  if(hist==NULL) return;

  hist->SetYTitle("Error Bit");
  hist->GetYaxis()->SetBinLabel(1,"Overflow Warn");
  hist->GetYaxis()->SetBinLabel(2,"Buffer Busy");
  hist->GetYaxis()->SetBinLabel(3,"Empty Event");
  hist->GetYaxis()->SetBinLabel(4,"Reject L1A");
  hist->GetYaxis()->SetBinLabel(5,"Latency Err");
  hist->GetYaxis()->SetBinLabel(6,"Latency Warn");
  hist->GetYaxis()->SetBinLabel(7,"OpDat Err");
  hist->GetYaxis()->SetBinLabel(8,"Clock Err");
  hist->GetYaxis()->SetBinLabel(9,"Bunch Err");
  hist->GetYaxis()->SetBinLabel(13,"Test Mode");
  hist->GetYaxis()->SetBinLabel(14,"Histo Mode");
  hist->GetYaxis()->SetBinLabel(15,"Calib Trig");
  hist->GetYaxis()->SetBinLabel(16,"Bit15 Err");
  
  return;
}


void HcalDataFormatClient::report(){
  if(!dbe_) return;
  if ( debug_ ) cout << "HcalDataFormatClient: report" << endl;
  
  char name[256];
  
  sprintf(name, "%sHcal/DataFormatMonitor/Data Format Task Event Number",process_.c_str());
  MonitorElement* me = dbe_->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_ ) cout << "Found '" << name << "'" << endl;
  }
  else printf("Didn't find %s\n",name);
  getHistograms();
  
  return;
}

void HcalDataFormatClient::resetAllME(){

  if(!dbe_) return;
  
  char name[150];     
  sprintf(name,"%sHcal/DataFormatMonitor/Spigot Format Errors",process_.c_str());
  resetME(name,dbe_);
  
  sprintf(name,"%sHcal/DataFormatMonitor/Num Bad Quality Digis -DV bit-Err bit-Cap Rotation",process_.c_str());
  resetME(name,dbe_);
  
  sprintf(name,"%sHcal/DataFormatMonitor/Num Unmapped Digis",process_.c_str());
  resetME(name,dbe_);
  
  sprintf(name,"%sHcal/DataFormatMonitor/Num Unmapped Trigger Primitive Digis",process_.c_str());
  resetME(name,dbe_);
  
  sprintf(name,"%sHcal/DataFormatMonitor/FED Error Map from Unpacker Report",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/BCN from HTRs",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/BCN from DCCs",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/BCN Inconsistent - HTR vs Ref HTR",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/EvN Inconsistent - HTR vs Ref HTR",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/FibOrbMsgBCN",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/EvN Difference Between Ref HTR and DCC",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/BCN Difference Between Ref HTR and DCC",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Firmware Version",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/Invalid HTR Data",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word by Crate",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 0",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 1",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 2",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 3",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 4",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 5",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 6",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 7",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 8",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 9",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 10",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 11",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 12",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 13",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 14",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 15",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 16",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sHcal/DataFormatMonitor/HTR Error Word - Crate 17",process_.c_str());
  resetME(name,dbe_);

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HBHE";
    if(i==1) type = "HBHE";
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";

    sprintf(name,"%sHcal/DataFormatMonitor/%s Data Format Error Word",process_.c_str(), type.c_str());
    resetME(name,dbe_);

  }
  
  return;
}

void HcalDataFormatClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  cout << "Preparing HcalDataFormatClient html output ..." << endl;
  string client = "DataFormatMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Data Format Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Data Format</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"DataFormatMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"DataFormatMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"DataFormatMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<h2><strong>Hcal DCC Error Word</strong></h2>" << endl;  
  htmlFile << "<h3>" << endl;
  if(subDetsOn_[0]||subDetsOn_[1]) htmlFile << "<a href=\"#HBHE_Plots\">HBHE Plots </a></br>" << endl;
  //if(subDetsOn_[1]) htmlFile << "<a href=\"#HBHE_Plots\">HBHE Plots </a></br>" << endl;
  if(subDetsOn_[2]) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  if(subDetsOn_[3]) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;

  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrMapbyCrate_,"Crate #"," ", 23, htmlFile,htmlDir);
  histoHTML(runNo,BCN_,"Bunch Counter Number","Events", 23, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,dccBCN_,"Bunch Counter Number","Events", 23, htmlFile,htmlDir);
  histoHTML2(runNo,InvHTRData_,"Spigot #","DCC #", 23, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,CDF_Violation_,"HCAL FED ID"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,DCC_Evt_Fmt_,"HCAL FED ID","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,DCC_Spigot_Err_,"HCAL FED ID","", 92, htmlFile,htmlDir);
  histoHTML2(runNo,DCC_Err_Warn_,"HCAL FED ID","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,BCNMap_,"Slot #","Crate #", 92, htmlFile,htmlDir);
  histoHTML2(runNo,EvtMap_,"Slot #","Crate #", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,BCNCheck_,"htr BCN - dcc BCN"," ", 92, htmlFile,htmlDir);
  histoHTML(runNo,EvtNCheck_,"htr Evt # - dcc Evt #","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
 
  htmlFile << "<tr align=\"center\">" << endl;
  histoHTML(runNo,FibOrbMsgBCN_,"Fiber Orbit Message BCN","Events", 30, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,unmappedDigis_,"# Digis","Events", 92, htmlFile,htmlDir);
  histoHTML(runNo,unmappedTPDs_,"# TP Digis","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,spigotErrs_,"# Errs","Events", 92, htmlFile,htmlDir);
  histoHTML(runNo,badDigis_,"# Bad Digis","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,fedErrMap_,"DCC Id","# Errors", 92, htmlFile,htmlDir);
  histoHTML2(runNo,FWVerbyCrate_,"Firmware Version","Crate #", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrCrate0_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,ErrCrate1_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrCrate2_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,ErrCrate3_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrCrate4_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,ErrCrate5_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

 htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrCrate6_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,ErrCrate7_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

 htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrCrate8_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,ErrCrate9_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

 htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrCrate10_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,ErrCrate11_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

 htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrCrate12_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,ErrCrate13_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

 htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrCrate14_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,ErrCrate15_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

 htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrCrate16_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,ErrCrate17_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;


  bool HBOn_ = subDetsOn_[0];
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    
    string type = "HBHE";
    if(i==1) type = "HBHE"; 
    else if(i==2) type = "HF"; 
    else if(i==3) type = "HO"; 
    if (i==1 && HBOn_) continue;
    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;
    htmlFile << "<tr align=\"left\">" << endl;
    int ind = i-1;
    if (ind<0) ind = 0;
    histoHTML(runNo,dferr_[ind],"Error Bit","Frequency", 92, htmlFile,htmlDir);
    htmlFile << "<tr align=\"left\">" << endl;
    /*
    histoHTML2(runNo,crateErrMap_[i],"VME Crate ID","HTR Slot", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;   
    htmlFile << "<tr align=\"left\">" << endl;  
    histoHTML2(runNo,spigotErrMap_[i],"Spigot","DCC Id", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
    */
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;   
  
  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();
   return;
}


void HcalDataFormatClient::createTests(){

  if(debug_) cout << "HcalDataFormatClient: creating tests" << endl;

  if(!dbe_) return;

  char meTitle[250], name[250];    
  vector<string> params;
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HBHE";
    if(i==1) type = "HBHE"; 
    else if(i==2) type = "HF"; 
    else if(i==3) type = "HO"; 
    
    sprintf(meTitle,"%sHcal/DataFormatMonitor/%s Data Format Error Word",process_.c_str(),type.c_str());
    sprintf(name,"DFMon %s HTR Err Word",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = dbe_->get(meTitle);
      if(me){
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("1.0"); params.push_back("0.95");  //warn, err probs
	params.push_back("0"); params.push_back("0");  //ymin, ymax
	createYRangeTest(dbe_, params);
      }
    }
  }

  sprintf(meTitle,"%sHcal/DataFormatMonitor/DCC Event Format violation",process_.c_str());
  sprintf(name,"DFMon DCC Evt Format");
  if(dqmQtests_.find(name) == dqmQtests_.end()){	
    MonitorElement* me = dbe_->get(meTitle);
    if(me){
      dqmQtests_[name]=meTitle;	  
      params.clear();
      params.push_back(meTitle); params.push_back(name);  //hist and test titles
      createH2ContentTest(dbe_, params);
    }
  }
  
  sprintf(meTitle,"%sHcal/DataFormatMonitor/HTR Error Word by Crate",process_.c_str());
  sprintf(name,"DFMon Err Wd by Crate");
  if(dqmQtests_.find(name) == dqmQtests_.end()){	
    MonitorElement* me = dbe_->get(meTitle);
    if(me){
      dqmQtests_[name]=meTitle;	  
      params.clear();
      params.push_back(meTitle); params.push_back(name);  //hist and test titles
      createH2ContentTest(dbe_, params);
    }
  }

  sprintf(meTitle,"%sHcal/DataFormatMonitor/Common Data Format violations",process_.c_str());
  sprintf(name,"DFMon CDF Violations");
  if(dqmQtests_.find(name) == dqmQtests_.end()){	
    MonitorElement* me = dbe_->get(meTitle);
    if(me){
      dqmQtests_[name]=meTitle;	  
      params.clear();
      params.push_back(meTitle); params.push_back(name);  //hist and test titles
      createH2ContentTest(dbe_, params);
    }
  }

  sprintf(meTitle,"%sHcal/DataFormatMonitor/DCC Error and Warning",process_.c_str());
  sprintf(name,"DFMon DCC Err/Warn");
  if(dqmQtests_.find(name) == dqmQtests_.end()){	
    MonitorElement* me = dbe_->get(meTitle);
    if(me){
      dqmQtests_[name]=meTitle;	  
      params.clear();
      params.push_back(meTitle); params.push_back(name);  //hist and test titles
      createH2ContentTest(dbe_, params);
    }
  }
  
  sprintf(meTitle,"%sHcal/DataFormatMonitor/DCC Nonzero Spigot Conditions",process_.c_str());
  sprintf(name,"DFMon DCC Spigot Err");
  if(dqmQtests_.find(name) == dqmQtests_.end()){	
    MonitorElement* me = dbe_->get(meTitle);
    if(me){
      dqmQtests_[name]=meTitle;	  
      params.clear();
      params.push_back(meTitle); params.push_back(name);  //hist and test titles
      createH2ContentTest(dbe_, params);
    }
  }
  
  sprintf(meTitle,"%sHcal/DataFormatMonitor/Num Bad Quality Digis -DV bit-Err bit-Cap Rotation",process_.c_str());
  sprintf(name,"DFMon # Bad Digis");
  if(dqmQtests_.find(name) == dqmQtests_.end()){	
    MonitorElement* me = dbe_->get(meTitle);
    if(me){	
      dqmQtests_[name]=meTitle;	  
      params.clear();
      params.push_back(meTitle); params.push_back(name);  //hist and test titles
      params.push_back("1.0"); params.push_back("0.99");  //warn, err probs
      params.push_back("0.0"); params.push_back("0.0");  //xmin, xmax
      createXRangeTest(dbe_, params);
    }
  }

  sprintf(meTitle,"%sHcal/DataFormatMonitor/Num Unmapped Digis"   ,process_.c_str()); 
  sprintf(name,"DFMon # Unmapped Digis");
  if(dqmQtests_.find(name) == dqmQtests_.end()){	
    MonitorElement* me = dbe_->get(meTitle);
    if(me){	
      dqmQtests_[name]=meTitle;	  
      params.clear();
      params.push_back(meTitle); params.push_back(name);  //hist and test titles
      params.push_back("1.0"); params.push_back("0.99");  //warn, err probs
      params.push_back("0.0"); params.push_back("0.0");  //xmin, xmax
      createXRangeTest(dbe_, params);
    }
  }

  sprintf(meTitle,"%sHcal/DataFormatMonitor/Num Unmapped Trigger Primitive Digis",process_.c_str()); 
  sprintf(name,"DFMon # Unmapped TP Digis");
  if(dqmQtests_.find(name) == dqmQtests_.end()){	
    MonitorElement* me = dbe_->get(meTitle);
    if(me){	
      dqmQtests_[name]=meTitle;	  
      params.clear();
      params.push_back(meTitle); params.push_back(name);  //hist and test titles
      params.push_back("1.0"); params.push_back("0.99");  //warn, err probs
      params.push_back("0.0"); params.push_back("0.0");  //xmin, xmax
      createXRangeTest(dbe_, params);
    }
  }
  
  return;
}

void HcalDataFormatClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/DataFormatMonitor/Data Format Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  char name[150]; 
  sprintf(name,"DQMData/Hcal/DataFormatMonitor/Num Bad Quality Digis -DV bit-Err bit-Cap Rotation");
  badDigis_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/Num Unmapped Digis");
  unmappedDigis_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/Num Unmapped Trigger Primitive Digis");
  unmappedTPDs_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/FED Error Map from Unpacker Report");
  fedErrMap_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/BCN from HTRs");
  BCN_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/BCN from DCCs");
  dccBCN_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/BCN Difference Between Ref HTR and DCC");
  BCNCheck_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/EvN Difference Between Ref HTR and DCC");
  EvtNCheck_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/BCN of Fiber Orbit Message");
  FibOrbMsgBCN_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/EvN Inconsistent - HTR vs Ref HTR");
  EvtMap_ = (TH2F*)infile->Get(name);
  
  sprintf(name,"DQMData/Hcal/DataFormatMonitor/BCN Inconsistent - HTR vs Ref HTR");
  BCNMap_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Firmware Version");
  FWVerbyCrate_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/Invalid HTR Data");
  InvHTRData_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word by Crate");
  ErrMapbyCrate_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 0");
  ErrCrate0_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 1");
  ErrCrate1_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 2");
  ErrCrate2_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 3");
  ErrCrate3_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 4");
  ErrCrate4_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 5");
  ErrCrate5_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 6");
  ErrCrate6_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 7");
  ErrCrate7_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 8");
  ErrCrate8_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 9");
  ErrCrate9_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 10");
  ErrCrate10_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 11");
  ErrCrate11_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 12");
  ErrCrate12_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 13");
  ErrCrate13_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 14");
  ErrCrate14_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 15");
  ErrCrate15_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 16");
  ErrCrate16_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Error Word - Crate 17");
  ErrCrate17_ = (TH2F*)infile->Get(name);

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HBHE";
    if(i==1) type = "HBHE";
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";

    sprintf(name,"DQMData/Hcal/DataFormatMonitor/%s Data Format Error Word", type.c_str());
    int ind = i-1;
    if (i<0) i=0;
    dferr_[ind] = (TH1F*)infile->Get(name);    
    labelxBits(dferr_[ind]);
    /*    
    sprintf(name,"DQMData/Hcal/DataFormatMonitor/%s Data Format Crate Error Map", type.c_str());
    crateErrMap_[i] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DataFormatMonitor/%s Data Format Spigot Error Map", type.c_str());
    spigotErrMap_[i] = (TH2F*)infile->Get(name);
    */
  }

  return;
}
