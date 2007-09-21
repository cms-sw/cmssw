#include <DQM/HcalMonitorClient/interface/HcalDataFormatClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>

HcalDataFormatClient::HcalDataFormatClient(const ParameterSet& ps, MonitorUserInterface* mui){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = mui;
  for(int i=0; i<4; i++){
    dferr_[i] = NULL;
    crateErrMap_[i] =NULL;
    spigotErrMap_[i] = NULL;
  }
  spigotErrs_ = NULL;
  badDigis_ = NULL;
  unmappedDigis_ = NULL;
  unmappedTPDs_ = NULL;
  fedErrMap_ = NULL;

  ievt_=0; jevt_=0;

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

HcalDataFormatClient::HcalDataFormatClient(){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  verbose_ =false;
  mui_ = 0;
  for(int i=0; i<4; i++){
    dferr_[i] = NULL;
    crateErrMap_[i] =NULL;
    spigotErrMap_[i] = NULL;
  }

  spigotErrs_ = NULL;
  badDigis_ = NULL;
  unmappedDigis_ = NULL;
  unmappedTPDs_ = NULL;
  fedErrMap_ = NULL;

  for(int i=0; i<4; i++) subDetsOn_[i] = false;
}

HcalDataFormatClient::~HcalDataFormatClient(){

  this->cleanup();
  
}

void HcalDataFormatClient::beginJob(void){

  if ( verbose_ ) cout << "HcalDataFormatClient: beginJob" << endl;
  ievt_ = 0; jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalDataFormatClient::beginRun(void){

  if ( verbose_ ) cout << "HcalDataFormatClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalDataFormatClient::endJob(void) {

  if ( verbose_ ) cout << "HcalDataFormatClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();
  return;
}

void HcalDataFormatClient::endRun(void) {

  if ( verbose_ ) cout << "HcalDataFormatClient: endRun, jevt = " << jevt_ << endl;

  //  this->resetME();
  //  this->unsubscribe();
  this->cleanup();
  return;
}

void HcalDataFormatClient::setup(void) {
  return;
}

void HcalDataFormatClient::cleanup(void) {

  if ( cloneME_ ) {
    for(int i=0; i<4; i++){
      if ( dferr_[i] ) delete dferr_[i];
      if ( crateErrMap_[i]) delete crateErrMap_[i];
      if ( spigotErrMap_[i]) delete spigotErrMap_[i];
    }
  
    if ( spigotErrs_) delete spigotErrs_;
    if ( badDigis_) delete badDigis_;
    if ( unmappedDigis_) delete unmappedDigis_;
    if ( unmappedTPDs_) delete unmappedTPDs_;
    if ( fedErrMap_) delete fedErrMap_;
  }  
  for(int i=0; i<4; i++){
    dferr_[i] = NULL;
    crateErrMap_[i] =NULL;
    spigotErrMap_[i] = NULL;
  }
  
  spigotErrs_ = NULL;
  badDigis_ = NULL;
  unmappedDigis_ = NULL;
  unmappedTPDs_ = NULL;
  fedErrMap_ = NULL;

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  return;
}

void HcalDataFormatClient::subscribe(void){

  if ( verbose_ ) cout << "HcalDataFormatClient: subscribe" << endl;
  if(mui_) mui_->subscribe("*/HcalMonitor/DataFormatMonitor/*");
  return;
}

void HcalDataFormatClient::subscribeNew(void){
  if(mui_) mui_->subscribeNew("*/HcalMonitor/DataFormatMonitor/*");
  return;
}

void HcalDataFormatClient::unsubscribe(void){

  if ( verbose_ ) cout << "HcalDataFormatClient: unsubscribe" << endl;
  if(mui_) mui_->unsubscribe("*/HcalMonitor/DataFormatMonitor/*");
  return;
}

void HcalDataFormatClient::errorOutput(){
  
  if(!mui_) return;

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  
  for (map<string, string>::iterator testsMap=dqmQtests_.begin(); testsMap!=dqmQtests_.end();testsMap++){
    string testName = testsMap->first;
    string meName = testsMap->second;
    MonitorElement* me = mui_->get(meName);
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
  printf("Data Format Task: %d errs, %d warnings, %d others\n",dqmReportMapErr_.size(),dqmReportMapWarn_.size(),dqmReportMapOther_.size());
  return;
}

void HcalDataFormatClient::getErrors(map<string, vector<QReport*> > outE, map<string, vector<QReport*> > outW, map<string, vector<QReport*> > outO){

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

void HcalDataFormatClient::analyze(void){

  jevt_++;
  int updates = 0;
  if(mui_) mui_->getNumUpdates();
  if ( updates % 10 == 0 ) {
    if ( verbose_ ) cout << "HcalDataFormatClient: " << updates << " updates" << endl;
  }

  return;
}

void HcalDataFormatClient::getHistograms(){
  char name[150];     
  sprintf(name,"DataFormatMonitor/Spigot Format Errors");
  spigotErrs_ = getHisto(name, process_, mui_, verbose_,cloneME_);

  sprintf(name,"DataFormatMonitor/Bad Quality Digis");
  badDigis_ = getHisto(name, process_, mui_, verbose_,cloneME_);

  sprintf(name,"DataFormatMonitor/Unmapped Digis");
  unmappedDigis_ = getHisto(name, process_, mui_, verbose_,cloneME_);

  sprintf(name,"DataFormatMonitor/Unmapped Trigger Primitive Digis");
  unmappedTPDs_ = getHisto(name, process_, mui_, verbose_,cloneME_);

  sprintf(name,"DataFormatMonitor/FED Error Map");
  fedErrMap_ = getHisto(name, process_, mui_, verbose_,cloneME_);
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE";
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";
    sprintf(name,"DataFormatMonitor/%s Data Format Error Words", type.c_str());
    dferr_[i] = getHisto(name, process_, mui_, verbose_,cloneME_);    
    labelBits(dferr_[i]);
    
    sprintf(name,"DataFormatMonitor/%s Data Format Crate Error Map", type.c_str());
    crateErrMap_[i] = getHisto2(name, process_, mui_, verbose_,cloneME_);

    sprintf(name,"DataFormatMonitor/%s Data Format Spigot Error Map", type.c_str());
    spigotErrMap_[i] = getHisto2(name, process_, mui_, verbose_,cloneME_);

  }
  return;
}


void HcalDataFormatClient::labelBits(TH1F* hist){
  
  if(hist==NULL) return;

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
  hist->GetXaxis()->SetBinLabel(10,"Test Mode");
  hist->GetXaxis()->SetBinLabel(11,"Histo Mode");
  hist->GetXaxis()->SetBinLabel(12,"Calib Trig");
  
  return;
}

void HcalDataFormatClient::report(){
  if(!mui_) return;
  if ( verbose_ ) cout << "HcalDataFormatClient: report" << endl;
  this->setup();
  
  char name[256];
  sprintf(name, "%sHcalMonitor/DataFormatMonitor/Data Format Task Event Number",process_.c_str());
  MonitorElement* me = mui_->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( verbose_ ) cout << "Found '" << name << "'" << endl;
  }

  getHistograms();
  
  return;
}

void HcalDataFormatClient::resetME(){

  if(!mui_) return;
  
  MonitorElement* me;
  char name[150];     

  sprintf(name,"%sHcalMonitor/HcalMonitor/DataFormatMonitor/Spigot Format Errors",process_.c_str());
  me = mui_->get(name);
  if(me) mui_->softReset(me);

  sprintf(name,"%sHcalMonitor/HcalMonitor/DataFormatMonitor/Bad Quality Digis",process_.c_str());
   me = mui_->get(name);
  if(me) mui_->softReset(me);

  sprintf(name,"%sHcalMonitor/HcalMonitor/DataFormatMonitor/Unmapped Digis",process_.c_str());
   me = mui_->get(name);
  if(me) mui_->softReset(me);

  sprintf(name,"%sHcalMonitor/HcalMonitor/DataFormatMonitor/Unmapped Trigger Primitive Digis",process_.c_str());
   me = mui_->get(name);
  if(me) mui_->softReset(me);

  sprintf(name,"%sHcalMonitor/HcalMonitor/DataFormatMonitor/FED Error Map",process_.c_str());
   me = mui_->get(name);
  if(me) mui_->softReset(me);

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE";
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";

    sprintf(name,"%sHcalMonitor/DataFormatMonitor/%s Data Format Error Words",process_.c_str(), type.c_str());
     me = mui_->get(name);
    if(me) mui_->softReset(me);

    sprintf(name,"%sHcalMonitor/DataFormatMonitor/%s Data Format Crate Error Map",process_.c_str(), type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);

    sprintf(name,"%sHcalMonitor/DataFormatMonitor/%s Data Format Spigot Error Map",process_.c_str(), type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);
    
  }
  
  return;
}

void HcalDataFormatClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing HcalDataFormatClient html output ..." << endl;
  string client = "DataFormatMonitor";
  htmlErrors(htmlDir,client,process_,mui_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);

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
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
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
  
  htmlFile << "<h2><strong>Hcal DCC Error Words</strong></h2>" << endl;  
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
  histoHTML(fedErrMap_,"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(spigotErrs_,"# Errs","Events", 92, htmlFile,htmlDir);
  histoHTML(badDigis_,"# Bad Digis","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(unmappedDigis_,"# Digis","Events", 92, htmlFile,htmlDir);
  histoHTML(unmappedTPDs_,"# TP Digis","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    
    string type = "HB";
    if(i==1) type = "HE"; 
    else if(i==2) type = "HF"; 
    else if(i==3) type = "HO"; 
    
    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;
    htmlFile << "<tr align=\"left\">" << endl;  
    histoHTML(dferr_[i],"Error Bit","Frequency", 92, htmlFile,htmlDir);
    histoHTML2(crateErrMap_[i],"VME Crate ID","HTR Slot", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;    
    htmlFile << "<tr align=\"left\">" << endl;  
    histoHTML2(spigotErrMap_[i],"Spigot","DCC Id", 100, htmlFile,htmlDir);
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


void HcalDataFormatClient::createTests(){
  if(!mui_) return;

  char meTitle[250], name[250];    
  vector<string> params;
  
  if(verbose_) printf("Creating Data Format tests...\n"); 
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    else if(i==2) type = "HF"; 
    else if(i==3) type = "HO";
    
    sprintf(meTitle,"%sHcalMonitor/DataFormatMonitor/%s Data Format Error Words",process_.c_str(),type.c_str());
    sprintf(name,"%s DataFormat",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = mui_->get(meTitle);
      if(me){
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("1.0"); params.push_back("0.95");  //warn, err probs
	params.push_back("0"); params.push_back("0");  //ymin, ymax
	createYRangeTest(mui_, params);
      }
    }
  }

  return;
}

void HcalDataFormatClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/DataFormatMonitor/Data Format Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }


  char name[150]; 

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/Spigot Format Errors");
  spigotErrs_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/Bad Quality Digis");
  badDigis_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/Unmapped Digis");
  unmappedDigis_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/Unmapped Trigger Primitive Digis");
  unmappedTPDs_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/FED Error Map");
  fedErrMap_ = (TH1F*)infile->Get(name);
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE";
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";

    sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/%s Data Format Error Words", type.c_str());
    dferr_[i] = (TH1F*)infile->Get(name);    
    labelBits(dferr_[i]);
    
    sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/%s Data Format Crate Error Map", type.c_str());
    crateErrMap_[i] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/%s Data Format Spigot Error Map", type.c_str());
    spigotErrMap_[i] = (TH2F*)infile->Get(name);

  }

  return;
}
