#include <DQM/HcalMonitorClient/interface/HcalDigiClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>

HcalDigiClient::HcalDigiClient(const ParameterSet& ps, MonitorUserInterface* mui){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = mui;
  for(int i=0; i<4; i++){
    gl_occ_geo_[i]=0;
    gl_err_geo_=0;
    if(i<3) gl_occ_elec_[i]=0;
    if(i<3) gl_err_elec_[i]=0;
    gl_occ_eta_ = 0;
    gl_occ_phi_ = 0;

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
    qie_capid_[i]=0; 
  }

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "HcalMonitor"); 

  vector<string> subdets = ps.getUntrackedParameter<vector<string> >("subDetsOn");
  for(int i=0; i<4; i++) subDetsOn_[i] = false;
  
  for(unsigned int i=0; i<subdets.size(); i++){
    if(subdets[i]=="HB") subDetsOn_[0] = true;
    else if(subdets[i]=="HE") subDetsOn_[1] = true;
    else if(subdets[i]=="HF") subDetsOn_[2] = true;
    else if(subdets[i]=="HO") subDetsOn_[3] = true;
  }

}

HcalDigiClient::HcalDigiClient(){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  mui_ = 0;
  for(int i=0; i<4; i++){
    gl_occ_geo_[i]=0;
    gl_err_geo_=0;
    if(i<3) gl_occ_elec_[i]=0;
    if(i<3) gl_err_elec_[i]=0;
    gl_occ_eta_ = 0;
    gl_occ_phi_ = 0;

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
    qie_capid_[i]=0; 
  }
  ievt_ = 0;
  jevt_ = 0;
  // verbosity switch
  verbose_ = false;
  
  for(int i=0; i<4; i++) subDetsOn_[i] = false;

}

HcalDigiClient::~HcalDigiClient(){

  this->cleanup();

}

void HcalDigiClient::beginJob(void){
  
  if ( verbose_ ) cout << "HcalDigiClient: beginJob" << endl;
  
  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalDigiClient::beginRun(void){

  if ( verbose_ ) cout << "HcalDigiClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalDigiClient::endJob(void) {

  if ( verbose_ ) cout << "HcalDigiClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup(); 
  return;
}

void HcalDigiClient::endRun(void) {

  if ( verbose_ ) cout << "HcalDigiClient: endRun, jevt = " << jevt_ << endl;

  //  this->resetME();
  //  this->unsubscribe();
  this->cleanup();  
  return;
}

void HcalDigiClient::setup(void) {
  
  return;
}

void HcalDigiClient::cleanup(void) {

  if ( cloneME_ ) {
    for(int i=0; i<4; i++){

      if(gl_occ_geo_[i]) delete gl_occ_geo_[i];
      if(gl_err_geo_) delete gl_err_geo_;
      if(i<3){
	if(gl_occ_elec_[i]) delete gl_occ_elec_[i];
	if(gl_err_elec_[i]) delete gl_err_elec_[i];
      }
      if(gl_occ_eta_) delete gl_occ_eta_;
      if(gl_occ_phi_) delete gl_occ_phi_;
      
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
      if(num_digi_[i]) delete num_digi_[i];      
    }    
  }
  for(int i=0; i<4; i++){
    gl_occ_geo_[i]=0;
    gl_err_geo_=0;
    if(i<3) gl_occ_elec_[i]=0;
    if(i<3) gl_err_elec_[i]=0;
    gl_occ_eta_ = 0;
    gl_occ_phi_ = 0;

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
    qie_capid_[i]=0; 
  }

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  return;
}

void HcalDigiClient::subscribe(void){

  if ( verbose_ ) cout << "HcalDigiClient: subscribe" << endl;
  if(mui_){
    mui_->subscribe("*/HcalMonitor/DigiMonitor/*");
    mui_->subscribe("*/HcalMonitor/DigiMonitor/HB/*");
    mui_->subscribe("*/HcalMonitor/DigiMonitor/HE/*");
    mui_->subscribe("*/HcalMonitor/DigiMonitor/HF/*");
    mui_->subscribe("*/HcalMonitor/DigiMonitor/HO/*");
  }
    return;
}

void HcalDigiClient::subscribeNew(void){
  if(mui_){
    mui_->subscribeNew("*/HcalMonitor/DigiMonitor/*");
    mui_->subscribeNew("*/HcalMonitor/DigiMonitor/HB/*");
    mui_->subscribeNew("*/HcalMonitor/DigiMonitor/HE/*");
    mui_->subscribeNew("*/HcalMonitor/DigiMonitor/HF/*");
    mui_->subscribeNew("*/HcalMonitor/DigiMonitor/HO/*");
  }
  return;
}

void HcalDigiClient::unsubscribe(void){

  if ( verbose_ ) cout << "HcalDigiClient: unsubscribe" << endl;
  if(mui_){
    mui_->unsubscribe("*/HcalMonitor/DigiMonitor/*");
    mui_->unsubscribe("*/HcalMonitor/DigiMonitor/HB/*");
    mui_->unsubscribe("*/HcalMonitor/DigiMonitor/HE/*");
    mui_->unsubscribe("*/HcalMonitor/DigiMonitor/HF/*");
    mui_->unsubscribe("*/HcalMonitor/DigiMonitor/HO/*");
  }
  return;
}

void HcalDigiClient::errorOutput(){
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  
  for (map<string, string>::iterator testsMap=dqmQtests_.begin(); testsMap!=dqmQtests_.end();testsMap++){
    string testName = testsMap->first;
    string meName = testsMap->second;
    MonitorElement* me = 0;
    if(mui_) me = mui_->get(meName);
    if(me){
      if (me->hasError()){
	vector<QReport*> report =  me->getQErrors();
	dqmReportMapErr_[meName] = report;
      }
      if (me->hasWarning()){
	vector<QReport*> report =  me->getQWarnings();
	dqmReportMapWarn_[meName] = report;
      }
      if(me->hasOtherReport()){
	vector<QReport*> report= me->getQOthers();
	dqmReportMapOther_[meName] = report;
      }
    }
  }
  printf("Digi Task: %d errors, %d warnings, %d others\n",dqmReportMapErr_.size(),dqmReportMapWarn_.size(),dqmReportMapOther_.size());

  return;
}

void HcalDigiClient::getErrors(map<string, vector<QReport*> > outE, map<string, vector<QReport*> > outW, map<string, vector<QReport*> > outO){

  this->errorOutput();
  outE.clear(); outW.clear(); outO.clear();

  for(map<string, vector<QReport*> >::iterator i=dqmReportMapErr_.begin(); i!=dqmReportMapErr_.end(); i++){
    outE[i->first] = i->second;
  }
  for(map<string, vector<QReport*> >::iterator i=dqmReportMapWarn_.begin(); i!=dqmReportMapWarn_.end(); i++){
    outW[i->first] = i->second;
  }
  for(map<string, vector<QReport*> >::iterator i=dqmReportMapOther_.begin(); i!=dqmReportMapOther_.end(); i++){
    outO[i->first] = i->second;
  }

  return;
}

void HcalDigiClient::report(){

  if ( verbose_ ) cout << "HcalDigiClient: report" << endl;
  //  this->setup();  
  
  char name[256];
  sprintf(name, "%sHcalMonitor/DigiMonitor/Digi Task Event Number",process_.c_str());
  MonitorElement* me = 0;
  if(mui_) me = mui_->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( verbose_ ) cout << "Found '" << name << "'" << endl;
  }

  getHistograms();

  return;
}

void HcalDigiClient::analyze(void){

  jevt_++;
  int updates = 0;
  if(mui_) mui_->getNumUpdates();
  if ( updates % 10 == 0 ) {
    if ( verbose_ ) cout << "HcalDigiClient: " << updates << " updates" << endl;
  }
  
  return;
}

void HcalDigiClient::getHistograms(){
  if(!mui_) return;

  char name[150];    
  sprintf(name,"DigiMonitor/Digi Geo Error Map");
  gl_err_geo_ = getHisto2(name, process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi VME Error Map");
  gl_err_elec_[0] = getHisto2(name,process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Fiber Error Map");
  gl_err_elec_[1] = getHisto2(name,process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Spigot Error Map");
  gl_err_elec_[2] = getHisto2(name,process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Depth 1 Occupancy Map");
  gl_occ_geo_[0] = getHisto2(name, process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Depth 2 Occupancy Map");
  gl_occ_geo_[1] = getHisto2(name, process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Depth 3 Occupancy Map");
  gl_occ_geo_[2] = getHisto2(name, process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Depth 4 Occupancy Map");
  gl_occ_geo_[3] = getHisto2(name, process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi VME Occupancy Map");
  gl_occ_elec_[0] = getHisto2(name,process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Fiber Occupancy Map");
  gl_occ_elec_[1] = getHisto2(name,process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Spigot Occupancy Map");
  gl_occ_elec_[2] = getHisto2(name,process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Eta Occupancy Map");
  gl_occ_eta_ = getHisto(name,process_, mui_,verbose_,cloneME_);
  
  sprintf(name,"DigiMonitor/Digi Phi Occupancy Map");
  gl_occ_phi_ = getHisto(name,process_, mui_,verbose_,cloneME_);
  
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE";
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 
    
    sprintf(name,"DigiMonitor/%s/%s Digi Geo Error Map",type.c_str(),type.c_str());
    sub_err_geo_[i] = getHisto2(name, process_, mui_,verbose_,cloneME_);
    
    sprintf(name,"DigiMonitor/%s/%s Digi VME Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][0] = getHisto2(name,process_, mui_,verbose_,cloneME_);
    
    sprintf(name,"DigiMonitor/%s/%s Digi Fiber Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][1] = getHisto2(name,process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Spigot Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][2] = getHisto2(name,process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Depth 1 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][0] = getHisto2(name, process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Depth 2 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][1] = getHisto2(name, process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Depth 3 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][2] = getHisto2(name, process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Depth 4 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][3] = getHisto2(name, process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi VME Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][0] = getHisto2(name,process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Fiber Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][1] = getHisto2(name,process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Spigot Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][2] = getHisto2(name,process_, mui_,verbose_,cloneME_);
    
    sprintf(name,"DigiMonitor/%s/%s Digi Eta Occupancy Map",type.c_str(),type.c_str());
    sub_occ_eta_[i] = getHisto(name,process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s Digi Phi Occupancy Map",type.c_str(),type.c_str());
    sub_occ_phi_[i] = getHisto(name,process_, mui_,verbose_,cloneME_);
    
    sprintf(name,"DigiMonitor/%s/%s QIE ADC Value",type.c_str(),type.c_str());
    qie_adc_[i] = getHisto(name, process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s # of Digis",type.c_str(),type.c_str());
    num_digi_[i] = getHisto(name, process_, mui_,verbose_,cloneME_);

    sprintf(name,"DigiMonitor/%s/%s QIE Cap-ID",type.c_str(),type.c_str());
    qie_capid_[i] = getHisto(name, process_, mui_,verbose_,cloneME_);
  }
  return;
}

void HcalDigiClient::resetME(){
  
  if(!mui_) return;

  Char_t name[150];    
  MonitorElement* me=0;

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 
    
    sprintf(name,"%sHcalMonitor/DigiMonitor/%s/%s Digi Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);

    sprintf(name,"%sHcalMonitor/DigiMonitor/%s/%s QIE ADC Value",process_.c_str(),type.c_str(),type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);
    sprintf(name,"%sHcalMonitor/DigiMonitor/%s/%s # of Digis",process_.c_str(),type.c_str(),type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);
    sprintf(name,"%sHcalMonitor/DigiMonitor/%s/%s QIE Cap-ID",process_.c_str(),type.c_str(),type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);
  }
  return;
}

void HcalDigiClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing HcalDigiClient html output ..." << endl;
  string client = "DigiMonitor";
  htmlErrors(htmlDir,client,process_,mui_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
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
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
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
  histoHTML2(gl_err_geo_,"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(gl_err_elec_[0],"VME Crate ID","HTR Slot", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(gl_err_elec_[1],"Fiber Channel","Fiber", 92, htmlFile,htmlDir);
  histoHTML2(gl_err_elec_[2],"Spigot","DCC Id", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(gl_occ_geo_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(gl_occ_geo_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(gl_occ_geo_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(gl_occ_geo_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(gl_occ_eta_,"iEta","Events", 92, htmlFile,htmlDir);
  histoHTML(gl_occ_phi_,"iPhi","Events", 100, htmlFile,htmlDir);
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
    histoHTML2(sub_err_geo_[i],"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML2(sub_err_elec_[i][0],"VME Crate ID","HTR Slot", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML2(sub_err_elec_[i][1],"Fiber Channel","Fiber", 92, htmlFile,htmlDir);
    histoHTML2(sub_err_elec_[i][2],"Spigot","DCC Id", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    int count = 0;
    htmlFile << "<tr align=\"left\">" << endl;	
    if(isValidGeom(i,0,0,1)){ histoHTML2(sub_occ_geo_[i][0],"iEta","iPhi", 92, htmlFile,htmlDir); count++; }
    if(isValidGeom(i,0,0,2)) { histoHTML2(sub_occ_geo_[i][1],"iEta","iPhi", 100, htmlFile,htmlDir); count++;}
    if(count%2==0){
      htmlFile << "</tr>" << endl;      
      htmlFile << "<tr align=\"left\">" << endl;	
    }
    if(isValidGeom(i,0,0,3)){histoHTML2(sub_occ_geo_[i][2],"iEta","iPhi", 92, htmlFile,htmlDir); count++;}
    if(count%2==0){
      htmlFile << "</tr>" << endl;      
      htmlFile << "<tr align=\"left\">" << endl;	
    }
    if(isValidGeom(i,0,0,4)){ histoHTML2(sub_occ_geo_[i][3],"iEta","iPhi", 100, htmlFile,htmlDir); count++;}
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(sub_occ_eta_[i],"iEta","Events", 92, htmlFile,htmlDir);
    histoHTML(sub_occ_phi_[i],"iPhi","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML2(sub_occ_elec_[i][0],"VME Crate ID","HTR Slot", 92, htmlFile,htmlDir);
    histoHTML2(sub_occ_elec_[i][1],"Fiber Channel","Fiber", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML2(sub_occ_elec_[i][2],"Spigot","DCC Id", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML(qie_adc_[i],"QIE ADC Value","Events", 92, htmlFile,htmlDir);
    histoHTML(qie_capid_[i],"QIE CAPID Value","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML(num_digi_[i],"Number of Digis","Events", 100, htmlFile,htmlDir);
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
  if(!mui_) return;

  char meTitle[250], name[250];    
  vector<string> params;
  
  if(verbose_) printf("Creating Digi tests...\n");
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;

    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO";
    
    sprintf(meTitle,"%sHcalMonitor/DigiMonitor/%s/%s Digi Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s Digi Errors by Geo_metry",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = mui_->get(meTitle);
      if(me){
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back((string)meTitle); params.push_back((string)name);  //hist and qtest titles
	params.push_back("0"); params.push_back("1e-10");  //mean ranges
	params.push_back("0"); params.push_back("1e-10");  //rms ranges
	createH2ContentTest(mui_, params);
      }
    }

    sprintf(meTitle,"%sHcalMonitor/DigiMonitor/%s/%s # of Digis",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s # of Digis",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = mui_->get(meTitle);
      if(me){	
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("1.0"); params.push_back("0.975");  //warn, err probs
	char high[20];	char low[20];
	sprintf(low,"%.2f\n", me->getMean());
	sprintf(high,"%.2f\n", me->getMean()+1);
	params.push_back(low); params.push_back(high);  //xmin, xmax
	createXRangeTest(mui_, params);
      }
    }

    sprintf(meTitle,"%sHcalMonitor/DigiMonitor/%s/%s QIE Cap-ID",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s QIE CapID",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = mui_->get(meTitle);
      if(me){	
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("1.0"); params.push_back("0.975");  //warn, err probs
	params.push_back("0"); params.push_back("3");  //xmin, xmax
	createXRangeTest(mui_, params);
      }
    }
    
  }

  return;
}

void HcalDigiClient::loadHistograms(TFile* infile){
  char name[150];    

  TNamed* tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/DigiMonitor/Digi Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Geo Error Map");
  gl_err_geo_ = (TH2F*)infile->Get(name);
  
  sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi VME Error Map");
  gl_err_elec_[0] = (TH2F*)infile->Get(name);
  
    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Fiber Error Map");
    gl_err_elec_[1] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Spigot Error Map");
    gl_err_elec_[2] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Depth 1 Occupancy Map");
    gl_occ_geo_[0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Depth 2 Occupancy Map");
    gl_occ_geo_[1] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Depth 3 Occupancy Map");
    gl_occ_geo_[2] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Depth 4 Occupancy Map");
    gl_occ_geo_[3] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Eta Occupancy Map");
    gl_occ_eta_ = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Phi Occupancy Map");
    gl_occ_phi_ = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi VME Occupancy Map");
    gl_occ_elec_[0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Fiber Occupancy Map");
    gl_occ_elec_[1] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/Digi Spigot Occupancy Map");
    gl_occ_elec_[2] = (TH2F*)infile->Get(name);


  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 


    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Geo Error Map",type.c_str(),type.c_str());
    sub_err_geo_[i] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi VME Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Fiber Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][1] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Spigot Error Map",type.c_str(),type.c_str());
    sub_err_elec_[i][2] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Depth 1 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Depth 2 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][1] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Depth 3 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][2] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Depth 4 Occupancy Map",type.c_str(),type.c_str());
    sub_occ_geo_[i][3] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Eta Occupancy Map",type.c_str(),type.c_str());
    sub_occ_eta_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Phi Occupancy Map",type.c_str(),type.c_str());
    sub_occ_phi_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi VME Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][0] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Fiber Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][1] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s Digi Spigot Occupancy Map",type.c_str(),type.c_str());
    sub_occ_elec_[i][2] = (TH2F*)infile->Get(name);
    
    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s QIE ADC Value",type.c_str(),type.c_str());
    qie_adc_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s # of Digis",type.c_str(),type.c_str());
    num_digi_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DigiMonitor/%s/%s QIE Cap-ID",type.c_str(),type.c_str());
    qie_capid_[i] = (TH1F*)infile->Get(name);
  }
  return;
}
