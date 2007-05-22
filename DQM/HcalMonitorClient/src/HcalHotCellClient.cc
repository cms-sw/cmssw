#include <DQM/HcalMonitorClient/interface/HcalHotCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>

HcalHotCellClient::HcalHotCellClient(const ParameterSet& ps, MonitorUserInterface* mui){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = mui;
  for(int i=0; i<4; i++){
    occ_geo[i]=0;
    occ_en[i]=0;
    max_en[i]=0;
    max_t[i]=0;
  }

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "HcalMonitor");

}

HcalHotCellClient::HcalHotCellClient(){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = 0;
  for(int i=0; i<4; i++){
    occ_geo[i]=0;
    occ_en[i]=0;
    max_en[i]=0;
    max_t[i]=0;
  }

  // verbosity switch
  verbose_ = false;

}

HcalHotCellClient::~HcalHotCellClient(){

  this->cleanup();

}

void HcalHotCellClient::beginJob(void){
  
  if ( verbose_ ) cout << "HcalHotCellClient: beginJob" << endl;
  
  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalHotCellClient::beginRun(void){

  if ( verbose_ ) cout << "HcalHotCellClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalHotCellClient::endJob(void) {

  if ( verbose_ ) cout << "HcalHotCellClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup(); 
  return;
}

void HcalHotCellClient::endRun(void) {

  if ( verbose_ ) cout << "HcalHotCellClient: endRun, jevt = " << jevt_ << endl;

  //  this->resetME();
  //  this->unsubscribe();
  this->cleanup();  
  return;
}

void HcalHotCellClient::setup(void) {
  
  return;
}

void HcalHotCellClient::cleanup(void) {

  if ( cloneME_ ) {
    for(int i=0; i<4; i++){
      if ( occ_geo[i]) delete occ_geo[i];  
      if ( occ_en[i]) delete occ_en[i];  
      if ( max_en[i]) delete max_en[i];  
      if ( max_t[i]) delete max_t[i];  

    }    
  }
  for(int i=0; i<4; i++){
    occ_geo[i]=0;
    occ_en[i]=0;
    max_en[i]=0;
    max_t[i]=0;
  }

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  return;
}

void HcalHotCellClient::subscribe(void){

  if ( verbose_ ) cout << "HcalHotCellClient: subscribe" << endl;
  if(mui_){
    mui_->subscribe("*/HcalMonitor/HotCellMonitor/*");
    mui_->subscribe("*/HcalMonitor/HotCellMonitor/HBHE/*");
    mui_->subscribe("*/HcalMonitor/HotCellMonitor/HF/*");
    mui_->subscribe("*/HcalMonitor/HotCellMonitor/HO/*");
  }
    return;
}

void HcalHotCellClient::subscribeNew(void){
  if(mui_){
    mui_->subscribeNew("*/HcalMonitor/HotCellMonitor/*");
    mui_->subscribeNew("*/HcalMonitor/HotCellMonitor/HBHE/*");
    mui_->subscribeNew("*/HcalMonitor/HotCellMonitor/HF/*");
    mui_->subscribeNew("*/HcalMonitor/HotCellMonitor/HO/*");
  }
  return;
}

void HcalHotCellClient::unsubscribe(void){

  if ( verbose_ ) cout << "HcalHotCellClient: unsubscribe" << endl;
  if(mui_){
    mui_->unsubscribe("*/HcalMonitor/HotCellMonitor/*");
    mui_->unsubscribe("*/HcalMonitor/HotCellMonitor/HBHE/*");
    mui_->unsubscribe("*/HcalMonitor/HotCellMonitor/HF/*");
    mui_->unsubscribe("*/HcalMonitor/HotCellMonitor/HO/*");
  }
  return;
}

void HcalHotCellClient::errorOutput(){
  
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
  printf("HotCell Task: %d errors, %d warnings, %d others\n",dqmReportMapErr_.size(),dqmReportMapWarn_.size(),dqmReportMapOther_.size());

  return;
}

void HcalHotCellClient::getErrors(map<string, vector<QReport*> > outE, map<string, vector<QReport*> > outW, map<string, vector<QReport*> > outO){

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

void HcalHotCellClient::report(){

  if ( verbose_ ) cout << "HcalHotCellClient: report" << endl;
  //  this->setup();  
  
  char name[256];
  sprintf(name, "Collector/%s/HcalMonitor/HotCellMonitor/HotCell Task Event Number",process_.c_str());
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

void HcalHotCellClient::analyze(void){

  jevt_++;
  int updates = mui_->getNumUpdates();
  if ( updates % 10 == 0 ) {
    if ( verbose_ ) cout << "HcalHotCellClient: " << updates << " updates" << endl;
  }
  
  return;
}

void HcalHotCellClient::getHistograms(){
  char name[150];    
  for(int i=0; i<4; i++){
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HO"; 
    if(i==3) type = "HF"; 

    sprintf(name,"HotCellMonitor/%s/%s HotCell Geo Occupancy Map",type.c_str(),type.c_str());
    occ_geo[i] = getHisto2(name, process_, mui_,verbose_,cloneME_);

    sprintf(name,"HotCellMonitor/%s/%s HotCell Geo Energy Map",type.c_str(),type.c_str());
    occ_en[i] = getHisto2(name, process_, mui_,verbose_,cloneME_);

    sprintf(name,"HotCellMonitor/%s/%s HotCell Energy",type.c_str(),type.c_str());
    max_en[i] = getHisto(name, process_, mui_,verbose_,cloneME_);

    sprintf(name,"HotCellMonitor/%s/%s HotCell Time",type.c_str(),type.c_str());
    max_t[i] = getHisto(name, process_, mui_,verbose_,cloneME_);
  }
  return;
}

void HcalHotCellClient::resetME(){
  
  Char_t name[150];    
  MonitorElement* me;

  for(int i=0; i<4; i++){
    string type = "HB";
    if(i==1) type = "HO"; 
    if(i==2) type = "HE"; 
    if(i==3) type = "HF"; 
    
    //    sprintf(name,"Collector/%s/HcalMonitor/HotCellMonitor/%s/%s HotCell Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    //    me = mui_->get(name);
    //    if(me) mui_->softReset(me);

  }
  return;
}

void HcalHotCellClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing HcalHotCellClient html output ..." << endl;
  string client = "HotCellMonitor";
  htmlErrors(htmlDir,client,process_,mui_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal HotCell Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal HotCells</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;

  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"HotCellMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"HotCellMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"HotCellMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal Hot Cell Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  htmlFile << "<a href=\"#HB_Plots\">HB Plots </a></br>" << endl;  htmlFile << "<a href=\"#HE_Plots\">HE Plots </a></br>" << endl;
  htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  
  for(int i=0; i<4; i++){
    htmlFile << "<tr align=\"left\">" << endl;
    
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HO"; 
    if(i==3) type = "HF"; 
    
    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML2(occ_geo[i],"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML2(occ_en[i],"iEta","iPhi", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML(max_en[i],"GeV","Evts", 92, htmlFile,htmlDir);
    histoHTML(max_t[i],"nS","Evts", 100, htmlFile,htmlDir);
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

void HcalHotCellClient::createTests(){
  char meTitle[250], name[250];    
  vector<string> params;
  
  printf("Creating Hot Cell tests...\n");
   
  return;
}

void HcalHotCellClient::loadHistograms(TFile* infile){
  char name[150];    

  TNamed* tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/HotCellMonitor/HotCell Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  for(int i=0; i<4; i++){
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HO"; 
    if(i==3) type = "HF"; 
    
    sprintf(name,"DQMData/HcalMonitor/HotCellMonitor/%s/%s HotCell Geo Occupancy Map",type.c_str(),type.c_str());
    occ_geo[i] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/HotCellMonitor/%s/%s HotCell Geo Energy Map",type.c_str(),type.c_str());
    occ_en[i] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/HotCellMonitor/%s/%s HotCell Energy",type.c_str(),type.c_str());
    max_en[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/HotCellMonitor/%s/%s HotCell Time",type.c_str(),type.c_str());
    max_t[i] = (TH1F*)infile->Get(name);

  }
  return;
}
