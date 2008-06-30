#include <DQM/HcalMonitorClient/interface/HcalTrigPrimClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalTrigPrimClient::HcalTrigPrimClient(){}


void HcalTrigPrimClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  for(int i=0; i<10; i++) tpSpectrum_[i] = NULL;

  tpCount_ = NULL;
  tpCountThr_ = NULL;
  tpSize_ = NULL;
  tpSpectrumAll_ = NULL;
  tpETSumAll_ = NULL;
  tpSOI_ET_ = NULL;
  OCC_ETA_ = NULL;
  OCC_PHI_ = NULL;
  OCC_ELEC_VME_ = NULL;
  OCC_ELEC_DCC_ = NULL;
  OCC_MAP_GEO_ = NULL;

  OCC_MAP_THR_ = NULL;
  EN_ETA_ = NULL;
  EN_PHI_ = NULL;
  EN_ELEC_VME_ = NULL;
  EN_ELEC_DCC_ = NULL;
  EN_MAP_GEO_ = NULL;

 TPTiming_ = NULL;
 TPTimingTop_ = NULL;
 TPTimingBot_ = NULL; 
 TPOcc_ = NULL;
 TP_ADC_ = NULL;
 MAX_ADC_ = NULL;
 TS_MAX_ = NULL;
 TPvsDigi_ = NULL;


}

HcalTrigPrimClient::~HcalTrigPrimClient(){
  this->cleanup();  
}

void HcalTrigPrimClient::beginJob(void){
  if ( debug_ ) cout << "HcalTrigPrimClient: beginJob" << endl;

  ievt_ = 0; jevt_ = 0;
  return;
}

void HcalTrigPrimClient::beginRun(void){
  if ( debug_ ) cout << "HcalTrigPrimClient: beginRun" << endl;

  jevt_ = 0;
  this->resetAllME();
  return;
}

void HcalTrigPrimClient::endJob(void) {
  if ( debug_ ) cout << "HcalTrigPrimClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

  return;
}

void HcalTrigPrimClient::endRun(void) {

  if ( debug_ ) cout << "HcalTrigPrimClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

  return;
}



void HcalTrigPrimClient::cleanup(void) {

  if ( cloneME_ ) {
    
    for(int i=0; i<10; i++) if(tpSpectrum_[i]) delete tpSpectrum_[i];
    
    if(tpCount_) delete tpCount_;
    if(tpCountThr_) delete tpCountThr_;
    if(tpSize_) delete tpSize_;
    if(tpSpectrumAll_) delete tpSpectrumAll_;
    if(tpETSumAll_) delete tpETSumAll_;
    if(tpSOI_ET_) delete tpSOI_ET_;
    if(OCC_ETA_) delete OCC_ETA_;
    if(OCC_PHI_) delete OCC_PHI_;
    if(OCC_ELEC_VME_) delete OCC_ELEC_VME_;
    if(OCC_ELEC_DCC_) delete OCC_ELEC_DCC_;
    if(OCC_MAP_GEO_) delete OCC_MAP_GEO_;

    if(OCC_MAP_THR_) delete OCC_MAP_THR_;
    if(EN_ETA_) delete EN_ETA_;
    if(EN_PHI_) delete EN_PHI_;
    if(EN_ELEC_VME_) delete EN_ELEC_VME_;
    if(EN_ELEC_DCC_) delete EN_ELEC_DCC_;
    if(EN_MAP_GEO_) delete EN_MAP_GEO_;

    if(TPTiming_) delete TPTiming_ ;
    if(TPTimingTop_) delete  TPTimingTop_;
    if(TPTimingBot_) delete  TPTimingBot_;
    if(TPOcc_) delete TPOcc_;
    if(TP_ADC_) delete  TP_ADC_;
    if(MAX_ADC_) delete  MAX_ADC_;
    if(TS_MAX_) delete  TS_MAX_;
    if(TPvsDigi_) delete TPvsDigi_;
  }  

  for(int i=0; i<10; i++) tpSpectrum_[i] = NULL;

  tpCount_ = NULL;
  tpCountThr_ = NULL;
  tpSize_ = NULL;
  tpSpectrumAll_ = NULL;
  tpETSumAll_ = NULL;
  tpSOI_ET_ = NULL;
  OCC_ETA_ = NULL;
  OCC_PHI_ = NULL;
  OCC_ELEC_VME_ = NULL;
  OCC_ELEC_DCC_ = NULL;
  OCC_MAP_GEO_ = NULL;
 
  OCC_MAP_THR_ = NULL;
  EN_ETA_ = NULL;
  EN_PHI_ = NULL;
  EN_ELEC_VME_ = NULL;
  EN_ELEC_DCC_ = NULL;
  EN_MAP_GEO_ = NULL;
  TPTiming_ = NULL;
  TPTimingTop_ = NULL;
  TPTimingBot_ = NULL; 
  TPOcc_ = NULL;
  TP_ADC_ = NULL;
  MAX_ADC_ = NULL;
  TS_MAX_ = NULL;
  TPvsDigi_ = NULL;

  return;
}



void HcalTrigPrimClient::analyze(void){
  jevt_++;

  int updates = 0;
  if ( updates % 10 == 0 ) {
    if ( debug_ ) cout << "HcalTrigPrimClient: " << updates << " updates" << endl;
  }

  return;
}

void HcalTrigPrimClient::getHistograms(){

  if(!dbe_) return;

  tpCount_ = getHisto("TrigPrimMonitor/# TP Digis", process_, dbe_, debug_,cloneME_);
  assert(tpCount_!=NULL);

  tpCountThr_ = getHisto("TrigPrimMonitor/# TP Digis over Threshold", process_, dbe_, debug_,cloneME_);
  tpSize_ = getHisto("TrigPrimMonitor/TP Size", process_, dbe_, debug_,cloneME_);  
  char name[150];      
  for (int i=0; i<10; i++) {
    sprintf(name,"TrigPrimMonitor/TP Spectrum sample %d",i);
    tpSpectrum_[i]= getHisto(name, process_, dbe_, debug_,cloneME_);
  }
  tpSpectrumAll_ = getHisto("TrigPrimMonitor/Full TP Spectrum", process_, dbe_, debug_,cloneME_);
  tpETSumAll_ = getHisto("TrigPrimMonitor/TP ET Sum", process_, dbe_, debug_,cloneME_);
  tpSOI_ET_ = getHisto("TrigPrimMonitor/TP SOI ET", process_, dbe_, debug_,cloneME_);  
  OCC_ETA_ = getHisto("TrigPrimMonitor/TrigPrim Eta Occupancy Map",process_, dbe_, debug_,cloneME_);  
  OCC_PHI_ = getHisto("TrigPrimMonitor/TrigPrim Phi Occupancy Map",process_, dbe_, debug_,cloneME_);  
  OCC_ELEC_VME_ = getHisto2("TrigPrimMonitor/TrigPrim VME Occupancy Map",process_, dbe_, debug_,cloneME_);  
  OCC_ELEC_DCC_ = getHisto2("TrigPrimMonitor/TrigPrim Spigot Occupancy Map",process_, dbe_, debug_,cloneME_);  
  OCC_MAP_GEO_ = getHisto2("TrigPrimMonitor/TrigPrim Geo Occupancy Map",process_, dbe_, debug_,cloneME_);  

  OCC_MAP_THR_ = getHisto2("TrigPrimMonitor/TrigPrim Geo Threshold Map",process_, dbe_, debug_,cloneME_);  
  EN_ETA_ = getHisto("TrigPrimMonitor/TrigPrim Eta Energy Map",process_, dbe_, debug_,cloneME_);  
  EN_PHI_ = getHisto("TrigPrimMonitor/TrigPrim Phi Energy Map",process_, dbe_, debug_,cloneME_);  
  EN_ELEC_VME_ = getHisto2("TrigPrimMonitor/TrigPrim VME Energy Map",process_, dbe_, debug_,cloneME_);  
  EN_ELEC_DCC_ = getHisto2("TrigPrimMonitor/TrigPrim Spigot Energy Map",process_, dbe_, debug_,cloneME_);  
  EN_MAP_GEO_ = getHisto2("TrigPrimMonitor/TrigPrim Geo Energy Map",process_, dbe_, debug_,cloneME_);  

  TPTiming_ = getHisto("TrigPrimMonitor/TP Timing",process_, dbe_, debug_,cloneME_);  
  TPTimingTop_ = getHisto("TrigPrimMonitor/TP Timing (Top wedges)",process_, dbe_, debug_,cloneME_);  
  TPTimingBot_ = getHisto("TrigPrimMonitor/TP Timing (Bottom wedges)",process_, dbe_, debug_,cloneME_);  
  TP_ADC_ = getHisto("TrigPrimMonitor/ADC spectrum positive TP",process_, dbe_, debug_,cloneME_);  
  MAX_ADC_ = getHisto("TrigPrimMonitor/Max ADC in TP",process_, dbe_, debug_,cloneME_);  
  TS_MAX_ = getHisto("TrigPrimMonitor/TS with max ADC",process_, dbe_, debug_,cloneME_);  
  TPOcc_ = getHisto2("TrigPrimMonitor/TP Occupancy",process_, dbe_, debug_,cloneME_);  
  TPvsDigi_ = getHisto2("TrigPrimMonitor/TP vs Digi",process_, dbe_, debug_,cloneME_);  


  return;
}

void HcalTrigPrimClient::report(){
  if(!dbe_) return;
  if ( debug_ ) cout << "HcalTrigPrimClient: report" << endl;
  
  char name[256];
  
  sprintf(name, "%sHcal/TrigPrimMonitor/TrigPrim Event Number",process_.c_str());
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

void HcalTrigPrimClient::resetAllME(){

  if(!dbe_) return;
  
  char name[150];     
  sprintf(name,"%sHcal/TrigPrimMonitor/# TP Digis",process_.c_str());
  resetME(name, dbe_);
  sprintf(name,"%sHcal/TrigPrimMonitor/# TP Digis over Threshold",process_.c_str());
  resetME(name, dbe_);
  sprintf(name,"%sHcal/TrigPrimMonitor/TP Size",process_.c_str());
  resetME(name, dbe_);  
  for (int i=0; i<10; i++) {
    sprintf(name,"%sHcal/TrigPrimMonitor/TP Spectrum sample %d",process_.c_str(),i);
    resetME(name, dbe_);
  }
  sprintf(name,"%sHcal/TrigPrimMonitor/Full TP Spectrum",process_.c_str());
  resetME(name, dbe_);
  sprintf(name,"%sHcal/TrigPrimMonitor/TP ET Sum",process_.c_str());
  resetME(name, dbe_);
  sprintf(name,"%sHcal/TrigPrimMonitor/TP SOI ET",process_.c_str());
  resetME(name, dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim Eta Occupancy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim Phi Occupancy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim VME Occupancy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim Spigot Occupancy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim Geo Occupancy Map",process_.c_str());
  resetME(name,dbe_);  

  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim Geo Threshold Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim Eta Energy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim Phi Energy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim VME Energy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim Spigot Energy Map",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TrigPrim Geo Energy Map",process_.c_str());
  resetME(name,dbe_);  

  sprintf(name,"%sHcal/TrigPrimMonitor/TP Timing",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TP Timing (Top wedges)",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TP Timing (Bottom Wedges)",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/ADC spectrum positive TP",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/Max ADC in TP",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TS with max ADC",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TP Occupancy",process_.c_str());
  resetME(name,dbe_);  
  sprintf(name,"%sHcal/TrigPrimMonitor/TP vs Digi",process_.c_str());
  resetME(name,dbe_);  

  return;
}

void HcalTrigPrimClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  cout << "Preparing HcalTrigPrimClient html output ..." << endl;
  string client = "TrigPrimMonitor";
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
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"TrigPrimMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"TrigPrimMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"TrigPrimMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;


  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,tpCount_,"# TP Digis"," ", 92, htmlFile,htmlDir);
  histoHTML(runNo,tpCountThr_,"# TP Digis"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,tpSpectrumAll_ ,"TP Energy"," ", 92, htmlFile,htmlDir);
  histoHTML(runNo,tpETSumAll_,"TP ET Sum"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,tpSOI_ET_,"TP Energy"," ", 92, htmlFile,htmlDir);
  histoHTML(runNo,tpSize_,"# Samples"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,OCC_MAP_GEO_,"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,EN_MAP_GEO_,"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,OCC_ETA_,"iEta"," ", 92, htmlFile,htmlDir);
  histoHTML(runNo,EN_ETA_,"iEta"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,OCC_PHI_,"iPhi"," ", 92, htmlFile,htmlDir);
  histoHTML(runNo,EN_PHI_,"iPhi"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,OCC_ELEC_VME_,"Slot","Crate Id", 92, htmlFile,htmlDir);
  histoHTML2(runNo,EN_ELEC_VME_,"Slot","Crate Id", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,OCC_ELEC_DCC_,"Spigot","DCC Id", 92, htmlFile,htmlDir);
  histoHTML2(runNo,EN_ELEC_DCC_,"Spigot","DCC Id", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;


  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,TPTiming_,"","time", 92, htmlFile,htmlDir);
  histoHTML(runNo,TP_ADC_,"","raw ADC", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,MAX_ADC_,"ADC","raw ADC", 92, htmlFile,htmlDir);
  histoHTML(runNo,TS_MAX_,"TS","num at TS", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,TPTimingTop_,"","time", 92, htmlFile,htmlDir);
  histoHTML(runNo,TPTimingBot_,"","time", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

 htmlFile << "<tr align=\"left\">" << endl;
 histoHTML2(runNo,TPOcc_,"iEta","iPhi", 92, htmlFile,htmlDir);  
 histoHTML2(runNo,TPvsDigi_,"Digi","TP", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
 
  histoHTML2(runNo,OCC_MAP_THR_,"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;



  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;   
  
  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();
   return;
}


void HcalTrigPrimClient::createTests(){

  if(debug_) cout << "HcalTrigPrimClient: creating tests" << endl;

  if(!dbe_) return;

  return;
}

void HcalTrigPrimClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/TrigPrimMonitor/TrigPrim Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  return;
}
