#include <DQM/HcalMonitorClient/interface/HcalDigiClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalDigiClient::HcalDigiClient(){}

void HcalDigiClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  for(int i=0; i<4; i++){
    gl_occ_geo_[i]=0;
    if(i<3) gl_occ_elec_[i]=0;
    if(i<3) gl_err_elec_[i]=0;

    sub_occ_geo_[i][0]=0;  sub_occ_geo_[i][1]=0;
    sub_occ_geo_[i][2]=0;  sub_occ_geo_[i][3]=0;
    sub_occ_elec_[i][0]=0;
    sub_occ_elec_[i][1]=0;
    sub_occ_elec_[i][2]=0;
    sub_occ_eta_[i] = 0;
    sub_occ_phi_[i] = 0;

    sub_err_geo_[i]=0;  
    sub_err_elec_[i][0]=0;
    sub_err_elec_[i][1]=0;
    sub_err_elec_[i][2]=0;
    qie_adc_[i]=0;  num_digi_[i]=0;
    qie_capid_[i]=0; qie_dverr_[i]=0;

    sub_num_bqdigi_[i] = 0;
    sub_bqdigi_frac_[i] = 0;
    sub_capid_t0_[i] = 0;
  }
    gl_err_geo_=0;
    gl_occ_eta_ = 0;
    gl_occ_phi_ = 0;

    gl_num_digi_ = 0;
    gl_num_bqdigi_ = 0;
    gl_bqdigi_frac_ = 0;
    gl_capid_t0_ = 0;
}

HcalDigiClient::~HcalDigiClient(){
  cleanup();
}

void HcalDigiClient::beginJob(void){
  
  if ( debug_ ) cout << "HcalDigiClient: beginJob" << endl;
  
  ievt_ = 0;
  jevt_ = 0;
  setup();
  resetAllME();
  return;
}

void HcalDigiClient::beginRun(void){

  if ( debug_ ) cout << "HcalDigiClient: beginRun" << endl;

  jevt_ = 0;
  setup();
  resetAllME();
  return;
}

void HcalDigiClient::endJob(void) {

  if ( debug_ ) cout << "HcalDigiClient: endJob, ievt = " << ievt_ << endl;

  cleanup(); 
  return;
}

void HcalDigiClient::endRun(void) {

  if ( debug_ ) cout << "HcalDigiClient: endRun, jevt = " << jevt_ << endl;

  cleanup();  
  return;
}

void HcalDigiClient::setup(void) {
  
  return;
}

void HcalDigiClient::cleanup(void) {

  if ( cloneME_ ) {

    if(gl_err_geo_) delete gl_err_geo_;
    if(gl_occ_eta_) delete gl_occ_eta_;
    if(gl_occ_phi_) delete gl_occ_phi_;

    if(gl_num_digi_) delete gl_num_digi_;
    if(gl_num_bqdigi_) delete gl_num_bqdigi_;
    if(gl_bqdigi_frac_) delete gl_bqdigi_frac_;
    if(gl_capid_t0_) delete gl_capid_t0_;
    
    for(int i=0; i<4; i++){
      if(gl_occ_geo_[i]) delete gl_occ_geo_[i];
      if(i<3){
	if(gl_occ_elec_[i]) delete gl_occ_elec_[i];
	if(gl_err_elec_[i]) delete gl_err_elec_[i];
      }
      
      if(sub_occ_geo_[i][0]) delete sub_occ_geo_[i][0];  
      if(sub_occ_geo_[i][1]) delete sub_occ_geo_[i][1];
      if(sub_occ_geo_[i][2]) delete sub_occ_geo_[i][2];  
      if(sub_occ_geo_[i][3]) delete sub_occ_geo_[i][3];
      if(sub_occ_elec_[i][0]) delete sub_occ_elec_[i][0];
      if(sub_occ_elec_[i][1]) delete sub_occ_elec_[i][1];
      if(sub_occ_elec_[i][2]) delete sub_occ_elec_[i][2];
      if(sub_occ_eta_[i]) delete sub_occ_eta_[i];
      if(sub_occ_phi_[i]) delete sub_occ_phi_[i];
      
      if(sub_err_geo_[i]) delete sub_err_geo_[i];  
      if(sub_err_elec_[i][0]) delete sub_err_elec_[i][0];
      if(sub_err_elec_[i][1]) delete sub_err_elec_[i][1];
      if(sub_err_elec_[i][2]) delete sub_err_elec_[i][2];

      if(qie_adc_[i]) delete qie_adc_[i];
      if(qie_capid_[i]) delete qie_capid_[i];
      if(qie_dverr_[i]) delete qie_dverr_[i];
      if(num_digi_[i]) delete num_digi_[i]; 

      if(sub_num_bqdigi_[i]) delete sub_num_bqdigi_[i];      
      if(sub_bqdigi_frac_[i]) delete sub_bqdigi_frac_[i];      
      if(sub_capid_t0_[i]) delete sub_capid_t0_[i];           
    }    
  }


  gl_err_geo_=0;
  gl_occ_eta_ = 0;
  gl_occ_phi_ = 0;

  gl_num_digi_ = 0;
  gl_num_bqdigi_ = 0;
  gl_bqdigi_frac_ = 0;
  gl_capid_t0_ = 0;
  
  for(int i=0; i<4; i++){
    gl_occ_geo_[i]=0;
    if(i<3) gl_occ_elec_[i]=0;
    if(i<3) gl_err_elec_[i]=0;

    sub_occ_geo_[i][0]=0;  sub_occ_geo_[i][1]=0;
    sub_occ_geo_[i][2]=0;  sub_occ_geo_[i][3]=0;
    sub_occ_elec_[i][0]=0;
    sub_occ_elec_[i][1]=0;
    sub_occ_elec_[i][2]=0;
    sub_occ_eta_[i] = 0;
    sub_occ_phi_[i] = 0;

    sub_err_geo_[i]=0;  
    sub_err_elec_[i][0]=0;
    sub_err_elec_[i][1]=0;
    sub_err_elec_[i][2]=0;
    qie_adc_[i]=0;  num_digi_[i]=0;
    qie_capid_[i]=0; qie_dverr_[i]=0;

    sub_num_bqdigi_[i] = 0;
    sub_bqdigi_frac_[i] = 0;
    sub_capid_t0_[i] = 0;
  }

  return;
}


void HcalDigiClient::report(){

  if ( debug_ ) cout << "HcalDigiClient: report" << endl;
  
  char name[256];
  sprintf(name, "%sHcal/DigiMonitor/Digi Task Event Number",process_.c_str());
  MonitorElement* me = 0;
  if(dbe_) me = dbe_->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_ ) cout << "Found '" << name << "'" << endl;
  }

  getHistograms();

  return;
}

void HcalDigiClient::analyze(void){

  jevt_++;
  int updates = 0;

  if ( updates % 10 == 0 ) {
    if ( debug_ ) cout << "HcalDigiClient: " << updates << " updates" << endl;
  }
  
  return;
}

void HcalDigiClient::getHistograms(){
  if(!dbe_) return;

  char name[150];    
  sprintf(name,"DigiMonitor/Digi Geo Error Map");
  gl_err_geo_ = getHisto2(name, process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi VME Error Map");
  gl_err_elec_[0] = getHisto2(name,process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Spigot Error Map");
  gl_err_elec_[2] = getHisto2(name,process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Depth 1 Occupancy Map");
  gl_occ_geo_[0] = getHisto2(name, process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Depth 2 Occupancy Map");
  gl_occ_geo_[1] = getHisto2(name, process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Depth 3 Occupancy Map");
  gl_occ_geo_[2] = getHisto2(name, process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Depth 4 Occupancy Map");
  gl_occ_geo_[3] = getHisto2(name, process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi VME Occupancy Map");
  gl_occ_elec_[0] = getHisto2(name,process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Spigot Occupancy Map");
  gl_occ_elec_[2] = getHisto2(name,process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Eta Occupancy Map");
  gl_occ_eta_ = getHisto(name,process_, dbe_,debug_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Phi Occupancy Map");
  gl_occ_phi_ = getHisto(name,process_, dbe_,debug_,cloneME_);

  sprintf(name,"DigiMonitor/Capid 1st Time Slice");
  gl_capid_t0_ = getHisto(name,process_, dbe_,debug_,cloneME_);

  sprintf(name,"DigiMonitor/# of Digis");
  gl_num_digi_ = getHisto(name,process_, dbe_,debug_,cloneME_);

  sprintf(name,"DigiMonitor/# Bad Qual Digis");
  gl_num_bqdigi_ = getHisto(name,process_, dbe_,debug_,cloneME_);

  sprintf(name,"DigiMonitor/Bad Digi Fraction");
  gl_bqdigi_frac_ = getHisto(name,process_, dbe_,debug_,cloneME_);
   
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE";
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 
    
    sprintf(name,"DigiMonitor/%s/%s Digi Geo Error Map",type.c_str(),type.c_str());
    sub_err_geo_[i] = getHisto2(name, process_, dbe_,debug_,cloneME_);
    
    sprintf(name,"DigiMonitor/%s/%s Digi VME Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][0] = getHisto2(name,process_, dbe_,debug_,cloneME_);
    
    sprintf(name,"DigiMonitor/%s/%s Digi Fiber Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][1] = getHisto2(name,process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Spigot Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][2] = getHisto2(name,process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Depth 1 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][0] = getHisto2(name, process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Depth 2 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][1] = getHisto2(name, process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Depth 3 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][2] = getHisto2(name, process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Depth 4 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][3] = getHisto2(name, process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi VME Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][0] = getHisto2(name,process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Fiber Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][1] = getHisto2(name,process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Spigot Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][2] = getHisto2(name,process_, dbe_,debug_,cloneME_);
    
    sprintf(name,"DigiMonitor/%s/%s Digi Eta Occupancy Map",type.c_str(),type.c_str());
    sub_occ_eta_[i] = getHisto(name,process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Phi Occupancy Map",type.c_str(),type.c_str());
    sub_occ_phi_[i] = getHisto(name,process_, dbe_,debug_,cloneME_);
    
    sprintf(name,"DigiMonitor/%s/%s QIE ADC Value",type.c_str(),type.c_str());
    qie_adc_[i] = getHisto(name, process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s # of Digis",type.c_str(),type.c_str());
    num_digi_[i] = getHisto(name, process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s QIE Cap-ID",type.c_str(),type.c_str());
    qie_capid_[i] = getHisto(name, process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s QIE Data Valid Err Bits",type.c_str(),type.c_str());
    qie_dverr_[i] = getHisto(name, process_, dbe_,debug_,cloneME_);
    qie_dverr_[i]->GetXaxis()->SetBinLabel(1,"Err=0 DV=0");
    qie_dverr_[i]->GetXaxis()->SetBinLabel(2,"Err=0 DV=1");
    qie_dverr_[i]->GetXaxis()->SetBinLabel(3,"Err=1 DV=0");
    qie_dverr_[i]->GetXaxis()->SetBinLabel(4,"Err=1 DV=1");

    sprintf(name,"DigiMonitor/%s/%s # Bad Qual Digis",type.c_str(),type.c_str());
    sub_num_bqdigi_[i] = getHisto(name, process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Bad Digi Fraction",type.c_str(),type.c_str());
    sub_bqdigi_frac_[i] = getHisto(name, process_, dbe_,debug_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Capid 1st Time Slice",type.c_str(),type.c_str());
    sub_capid_t0_[i] = getHisto(name, process_, dbe_,debug_,cloneME_);



  }
  return;
}

void HcalDigiClient::resetAllME(){
  
  if(!dbe_) return;

  Char_t name[150];    
  
  for(int i=1; i<5; i++){
    sprintf(name,"%sHcal/DigiMonitor/Digi Depth %d Occupancy Map",process_.c_str(),i);
    resetME(name,dbe_);
  }
  sprintf(name,"%sHcal/DigiMonitor/Digi Eta Occupancy Map",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/DigiMonitor/Digi Phi Occupancy Map",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/DigiMonitor/Digi VME Occupancy Map",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/DigiMonitor/Digi Spigot Occupancy Map",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/DigiMonitor/Digi Geo Error Map",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/DigiMonitor/Digi VME Error Map",process_.c_str());
  resetME(name,dbe_);
  sprintf(name,"%sHcal/DigiMonitor/Digi Spigot Error Map",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sDigiMonitor/Capid 1st Time Slice",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sDigiMonitor/# of Digis",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sDigiMonitor/# Bad Qual Digis",process_.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sDigiMonitor/Bad Digi Fraction",process_.c_str());
  resetME(name,dbe_);
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 

    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Shape",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Shape - over thresh",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s # of Digis",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Size",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Presamples",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s QIE Cap-ID",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s QIE ADC Value",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s QIE Data Valid Err Bits",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi VME Error Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Fiber Error Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Spigot Error Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    for(int j=1; j<5; j++){
      sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Depth %d Occupancy Map",process_.c_str(),type.c_str(),type.c_str(),j);
      resetME(name,dbe_);
    }
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Eta Occupancy Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Phi Occupancy Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi VME Occupancy Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Fiber Occupancy Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/DigiMonitor/%s/%s Digi Spigot Occupancy Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);

    sprintf(name,"%sDigiMonitor/%s/%s Capid 1st Time Slice",process_.c_str(),type.c_str(),type.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sDigiMonitor/%s/%s # of Digis",process_.c_str(),type.c_str(),type.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sDigiMonitor/%s/%s # Bad Qual Digis",process_.c_str(),type.c_str(),type.c_str());
  resetME(name,dbe_);

  sprintf(name,"%sDigiMonitor/%s/%s Bad Digi Fraction",process_.c_str(),type.c_str(),type.c_str());
  resetME(name,dbe_);

  }
  return;
}

void HcalDigiClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  cout << "Preparing HcalDigiClient html output ..." << endl;
  string client = "DigiMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Digi Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Digis</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table width=100%  border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"DigiMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"DigiMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"DigiMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal Digi Histograms</strong></h2>" << endl;
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
  histoHTML(runNo,gl_num_digi_,"# Digis","Events", 92, htmlFile,htmlDir);
  histoHTML(runNo,gl_bqdigi_frac_,"Fraction bad quality digis","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,gl_occ_geo_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,gl_occ_geo_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,gl_occ_geo_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,gl_occ_geo_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,gl_occ_eta_,"iEta","Events", 92, htmlFile,htmlDir);
  histoHTML(runNo,gl_occ_phi_,"iPhi","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,gl_occ_elec_[0],"HTR Slot","VME Crate Id", 92, htmlFile,htmlDir);
  histoHTML2(runNo,gl_occ_elec_[2],"Spigot","DCC Id", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,gl_num_bqdigi_,"# Bad Quality Digis","Events", 92, htmlFile,htmlDir);
  histoHTML(runNo,gl_capid_t0_,"CapId T0 relative to 1st CapId","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,gl_err_geo_,"iEta","iPhi", 92, htmlFile,htmlDir);

  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,gl_err_elec_[0],"HTR Slot","VME Crate Id", 92, htmlFile,htmlDir);
  histoHTML2(runNo,gl_err_elec_[2],"Spigot","DCC Id", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 

    htmlFile << "<tr align=\"left\">" << endl;
    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML(runNo,num_digi_[i],"Number of Digis","Events", 92, htmlFile,htmlDir);
    histoHTML(runNo,sub_capid_t0_[i],"CapId (T0) - 1st CapId (T0)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    int count = 0;
    htmlFile << "<tr align=\"left\">" << endl;	
    if(isValidGeom(i,0,0,1)){ histoHTML2(runNo,sub_occ_geo_[i][0],"iEta","iPhi", 92, htmlFile,htmlDir); count++; }
    if(isValidGeom(i,0,0,2)) { histoHTML2(runNo,sub_occ_geo_[i][1],"iEta","iPhi", 100, htmlFile,htmlDir); count++;}
    if(count%2==0){
      htmlFile << "</tr>" << endl;      
      htmlFile << "<tr align=\"left\">" << endl;	
    }
    if(isValidGeom(i,0,0,3)){histoHTML2(runNo,sub_occ_geo_[i][2],"iEta","iPhi", 92, htmlFile,htmlDir); count++;}
    if(count%2==0){
      htmlFile << "</tr>" << endl;      
      htmlFile << "<tr align=\"left\">" << endl;	
    }
    if(isValidGeom(i,0,0,4)){ histoHTML2(runNo,sub_occ_geo_[i][3],"iEta","iPhi", 100, htmlFile,htmlDir); count++;}
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(runNo,sub_occ_eta_[i],"iEta","Events", 92, htmlFile,htmlDir);
    histoHTML(runNo,sub_occ_phi_[i],"iPhi","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML2(runNo,sub_occ_elec_[i][0],"HTR Slot", "VME Crate ID", 92, htmlFile,htmlDir);    histoHTML2(runNo,sub_occ_elec_[i][2],"Spigot","DCC Id", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(runNo,sub_num_bqdigi_[i],"# Bad Quality Digis","Events", 92, htmlFile,htmlDir);
    histoHTML(runNo,sub_bqdigi_frac_[i],"Bad Quality Digi Fraction","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML2(runNo,sub_err_geo_[i],"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML2(runNo,sub_err_elec_[i][0],"HTR Slot","VME Crate ID", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML2(runNo,sub_err_elec_[i][2],"Spigot","DCC Id", 92, htmlFile,htmlDir);
    histoHTML(runNo,qie_adc_[i],"QIE ADC Value","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML(runNo,qie_dverr_[i],"QIE Data Valid and Err Bits","Events", 92, htmlFile,htmlDir);	
    histoHTML(runNo,qie_capid_[i],"QIE CAPID Value","Events", 100, htmlFile,htmlDir);
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

void HcalDigiClient::createTests(){
  if(!dbe_) return;

  char meTitle[250], name[250];    
  vector<string> params;
  
  if(debug_) printf("Creating Digi tests...\n");
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;

    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO";
    
    sprintf(meTitle,"%sHcal/DigiMonitor/%s/%s Digi Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s Digi Errors by Geo_metry",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = dbe_->get(meTitle);
      if(me){
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back((string)meTitle); params.push_back((string)name);  //hist and qtest titles
	params.push_back("0"); params.push_back("1e-10");  //mean ranges
	params.push_back("0"); params.push_back("1e-10");  //rms ranges
	createH2ContentTest(dbe_, params);
      }
    }
    /*
    sprintf(meTitle,"%sHcal/DigiMonitor/%s/%s # of Digis",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s # of Digis",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = dbe_->get(meTitle);
      if(me){	
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("1.0"); params.push_back("0.975");  //warn, err probs
	char high[20];	char low[20];
	//Window below has problems with bin edge effects; should fix this.
	sprintf(low,"%.2f", me->getMean());
	sprintf(high,"%.2f", me->getMean()+1);
	params.push_back(low); params.push_back(high);  //xmin, xmax
	createXRangeTest(dbe_, params);
      }
    }
    */
    sprintf(meTitle,"%sHcal/DigiMonitor/%s/%s QIE Cap-ID",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s QIE CapID",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = dbe_->get(meTitle);
      if(me){	
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("1.0"); params.push_back("0.975");  //warn, err probs
	params.push_back("0"); params.push_back("3");  //xmin, xmax
	createXRangeTest(dbe_, params);
      }
    }

    sprintf(meTitle,"%sHcal/DigiMonitor/%s/%s QIE Data Valid Err Bits",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s DVErr",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = dbe_->get(meTitle);
      if(me){	
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("1.0"); params.push_back("0.975");  //warn, err probs
	params.push_back("0.9"); params.push_back("1.1");  //xmin, xmax
	createXRangeTest(dbe_, params);
      }
    }
    
  }

  return;
}

void HcalDigiClient::loadHistograms(TFile* infile){
  char name[150];    

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/DigiMonitor/Digi Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  sprintf(name,"DQMData/Hcal/DigiMonitor/Digi Geo Error Map");
  gl_err_geo_ = (TH2F*)infile->Get(name);
  
  sprintf(name,"DQMData/Hcal/DigiMonitor/Digi VME Error Map");
  gl_err_elec_[0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/Digi Spigot Error Map");
    gl_err_elec_[2] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/Digi Depth 1 Occupancy Map");
    gl_occ_geo_[0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/Digi Depth 2 Occupancy Map");
    gl_occ_geo_[1] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/Digi Depth 3 Occupancy Map");
    gl_occ_geo_[2] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/Digi Depth 4 Occupancy Map");
    gl_occ_geo_[3] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/Digi Eta Occupancy Map");
    gl_occ_eta_ = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/DigiMonitor/Digi Phi Occupancy Map");
    gl_occ_phi_ = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/Digi VME Occupancy Map");
    gl_occ_elec_[0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/Digi Spigot Occupancy Map");
    gl_occ_elec_[2] = (TH2F*)infile->Get(name);

    sprintf(name,"DigiMonitor/Capid 1st Time Slice");
    gl_capid_t0_ =  (TH1F*)infile->Get(name);

    sprintf(name,"DigiMonitor/# of Digis");
    gl_num_digi_ =  (TH1F*)infile->Get(name);

    sprintf(name,"DigiMonitor/# Bad Qual Digis");
    gl_num_bqdigi_ =  (TH1F*)infile->Get(name);

    sprintf(name,"DigiMonitor/Bad Digi Fraction");
    gl_bqdigi_frac_ =  (TH1F*)infile->Get(name);
   
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 


    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Geo Error Map",type.c_str(),type.c_str());
    sub_err_geo_[i] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi VME Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Fiber Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][1] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Spigot Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][2] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Depth 1 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Depth 2 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][1] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Depth 3 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][2] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Depth 4 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][3] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Eta Occupancy Map",type.c_str(),type.c_str());
    sub_occ_eta_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Phi Occupancy Map",type.c_str(),type.c_str());
    sub_occ_phi_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi VME Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Fiber Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][1] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Digi Spigot Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][2] = (TH2F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s QIE ADC Value",type.c_str(),type.c_str());
    qie_adc_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s # of Digis",type.c_str(),type.c_str());
    num_digi_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s QIE Cap-ID",type.c_str(),type.c_str());
    qie_capid_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s QIE Data Valid Err Bits",type.c_str(),type.c_str());
    qie_dverr_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Capid 1st Time Slice",type.c_str(),type.c_str());
    sub_capid_t0_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s # of Digis",type.c_str(),type.c_str());
    num_digi_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s # Bad Qual Digis",type.c_str(),type.c_str());
    sub_num_bqdigi_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/DigiMonitor/%s/%s Bad Digi Fraction",type.c_str(),type.c_str());
    sub_bqdigi_frac_[i] = (TH1F*)infile->Get(name);

  }
  return;
}
