#include <DQM/HcalMonitorClient/interface/HcalTBClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>

HcalTBClient::HcalTBClient(const ParameterSet& ps, MonitorUserInterface* mui){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = mui;
  for(int i=0; i<3; i++){
    CHK[i] = 0;
    TOFT_S[i] = 0;
    TOFT_J[i] = 0;
    TOF_DT[i] = 0;
    HTIME[i] = 0;
    HRES[i] = 0;
    HPHASE[i] = 0;
    ERES[i] = 0;
  }
  for(int i=0; i<8; i++){
    WC[i]=0;
    WCX[i]=0;
    WCY[i]=0;
  }

  TOFQ[0] = 0;  TOFQ[1] = 0;
  TRIGT= 0;
  L1AT= 0;
  BCT= 0;
  PHASE= 0;


  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "HcalMonitor");
}


HcalTBClient::HcalTBClient(){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  verbose_ =false;

  mui_ = 0;
  for(int i=0; i<3; i++){
    CHK[i] = 0;
    TOFT_S[i] = 0;
    TOFT_J[i] = 0;
    TOF_DT[i] = 0;
    HTIME[i] = 0;
    HRES[i] = 0;
    HPHASE[i] = 0;
    ERES[i] = 0;
  }
  for(int i=0; i<8; i++){
    WC[i]=0;
    WCX[i]=0;
    WCY[i]=0;
  }

  TOFQ[0] = 0;  TOFQ[1] = 0;
  TRIGT= 0;
  L1AT= 0;
  BCT= 0;
  PHASE= 0;

}

HcalTBClient::~HcalTBClient(){

  this->cleanup();

}

void HcalTBClient::beginJob(void){

  if ( verbose_ ) cout << "HcalTBClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalTBClient::beginRun(void){

  if ( verbose_ ) cout << "HcalTBClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalTBClient::endJob(void) {

  if ( verbose_ ) cout << "HcalTBClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();
  return;
}

void HcalTBClient::endRun(void) {

  if ( verbose_ ) cout << "HcalTBClient: endRun, jevt = " << jevt_ << endl;
  //  this->resetME();
  //  this->unsubscribe();
  this->cleanup();
  return;
}

void HcalTBClient::setup(void) {
  return;
}

void HcalTBClient::cleanup(void) {

  if( cloneME_ ){
    for(int i=0; i<3; i++){
      if ( CHK[i] )  delete CHK[i];
      if ( TOFT_S[i] )  delete TOFT_S[i];
      if ( TOFT_J[i] )  delete TOFT_J[i];
      if ( TOF_DT[i] ) delete TOF_DT[i];
      if ( HTIME[i] ) delete HTIME[i];
      if ( HRES[i] ) delete HRES[i];
      if ( HPHASE[i] ) delete HPHASE[i];
      if ( ERES[i] ) delete ERES[i];
    }    
    for(int i=0; i<8; i++){
      if ( WC[i] ) delete WC[i];
      if ( WCX[i] ) delete WCX[i];
      if ( WCY[i] ) delete WCY[i];
    }    
    if(TOFQ[0]) delete TOFQ[0];
    if(TOFQ[1]) delete TOFQ[1];
    if ( TRIGT) delete TRIGT;
    if ( L1AT) delete L1AT;
    if ( BCT) delete BCT;
    if ( PHASE) delete PHASE;
  }
  
  for(int i=0; i<3; i++){
    CHK[i] = 0;
    TOFT_S[i] = 0;
    TOFT_J[i] = 0;
    TOF_DT[i] = 0;
    HTIME[i] = 0;
    HRES[i] = 0;
    HPHASE[i] = 0;
    ERES[i] = 0;
  }
  for(int i=0; i<8; i++){
    WC[i]=0;
    WCX[i]=0;
    WCY[i]=0;
  }

  TOFQ[0] = 0;  TOFQ[1] = 0;
  TRIGT= 0;
  L1AT= 0;
  BCT= 0;
  PHASE= 0;
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  return;
}

void HcalTBClient::subscribe(void){

  if ( verbose_ ) cout << "HcalTBClient: subscribe" << endl;
  if(mui_){
    mui_->subscribe("*/TBMonitor/*");
    mui_->subscribe("*/TBMonitor/QADCMonitor/*");
    mui_->subscribe("*/TBMonitor/TimingMonitor/*");
    mui_->subscribe("*/TBMonitor/EventPositionMonitor/*");
  }
  return;
}

void HcalTBClient::subscribeNew(void){
   if(mui_){
     mui_->subscribeNew("*/TBMonitor/*");
     mui_->subscribeNew("*/TBMonitor/QADCMonitor/*");
     mui_->subscribeNew("*/TBMonitor/TimingMonitor/*");
     mui_->subscribeNew("*/TBMonitor/EventPositionMonitor/*");
   }
  return;
}

void HcalTBClient::unsubscribe(void){

  if ( verbose_ ) cout << "HcalTBClient: unsubscribe" << endl;
   if(mui_){
     mui_->unsubscribe("*/TBMonitor/*");
     mui_->unsubscribe("*/TBMonitor/QADCMonitor/*");
     mui_->unsubscribe("*/TBMonitor/TimingMonitor/*");
     mui_->unsubscribe("*/TBMonitor/EventPositionMonitor/*");
   }
  return;
}

void HcalTBClient::errorOutput(){
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  if(!mui_) return;
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
  printf("Test Beam Task: %d errs, %d warnings, %d others\n",dqmReportMapErr_.size(),dqmReportMapWarn_.size(),dqmReportMapOther_.size());

  return;
}

void HcalTBClient::getErrors(map<string, vector<QReport*> > outE, map<string, vector<QReport*> > outW, map<string, vector<QReport*> > outO){

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

void HcalTBClient::analyze(void){

  jevt_++;
  int updates = mui_->getNumUpdates();
  if ( updates % 10 == 0 ) {
    if ( verbose_ ) cout << "HcalTBClient: " << updates << " updates" << endl;
  }
  return;
}

void HcalTBClient::getHistograms(){
  char name[150];     
  MonitorElement* me=0;
  for(int i=0; i<3; i++){
    sprintf(name,"Collector/%s/TBMonitor/QADCMonitor/Cherenkov QADC %d", process_.c_str(),i+1);
    me = mui_->get(name);
    CHK[i] = getHisto(me, verbose_,cloneME_);
  }
  for(int i=0; i<2; i++){
    sprintf(name,"Collector/%s/TBMonitor/QADCMonitor/TOF QADC %d", process_.c_str(),i+1);
    me = mui_->get(name);
    TOFQ[i] = getHisto(me, verbose_,cloneME_);
                                                   
    sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/TOF TDC %d - Saleve", process_.c_str(),i+1);
    me = mui_->get(name);
    TOFT_S[i] = getHisto(me, verbose_,cloneME_);

    sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/TOF TDC %d - Jura", process_.c_str(),i+1);
    me = mui_->get(name);
    TOFT_J[i] = getHisto(me, verbose_,cloneME_);

  }
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/TOF Time - Saleve", process_.c_str());
  me = mui_->get(name);
  TOFT_S[2] = getHisto(me, verbose_,cloneME_);

  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/TOF Time - Jura", process_.c_str());
  me = mui_->get(name);
  TOFT_J[2] = getHisto(me, verbose_,cloneME_);

  int i = 0;
  for(char c = 'A'; c<='H'; c++){
    sprintf(name,"Collector/%s/TBMonitor/EventPositionMonitor/Wire Chamber %c Hits", process_.c_str(),c);
    me = mui_->get(name);
    WC[i] = getHisto2(me, verbose_,cloneME_);

    sprintf(name,"Collector/%s/TBMonitor/EventPositionMonitor/Wire Chamber %c X Hits", process_.c_str(),c);
    me = mui_->get(name);
    WCX[i] = getHisto(me, verbose_,cloneME_);

    sprintf(name,"Collector/%s/TBMonitor/EventPositionMonitor/Wire Chamber %c Y Hits", process_.c_str(),c);
    me = mui_->get(name);
    WCY[i] = getHisto(me, verbose_,cloneME_);
    i++;
  }
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/TOF TDC - Delta 1", process_.c_str());;
  me = mui_->get(name);
  TOF_DT[0] = getHisto(me, verbose_,cloneME_);
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/TOF TDC - Delta 2", process_.c_str());;
  me = mui_->get(name);
  TOF_DT[1] = getHisto(me, verbose_,cloneME_);

  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/TOF Time - Delta", process_.c_str());;
  me = mui_->get(name);
  TOF_DT[2] = getHisto(me, verbose_,cloneME_);

  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/Trigger Timing", process_.c_str());;
  me = mui_->get(name);
  TRIGT= getHisto(me, verbose_,cloneME_);
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/TTC L1A Timing", process_.c_str());;
  me = mui_->get(name);
  L1AT= getHisto(me, verbose_,cloneME_);
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/Beam Coincidence Timing", process_.c_str());;
  me = mui_->get(name);
  BCT= getHisto(me, verbose_,cloneME_);
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/TB Phase", process_.c_str());;
  me = mui_->get(name);
  PHASE= getHisto(me, verbose_,cloneME_);
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/HB Time", process_.c_str());;
  me = mui_->get(name);
  HTIME[0]= getHisto(me, verbose_,cloneME_);
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/HB Time Resolution", process_.c_str());;
  me = mui_->get(name);
  HRES[0]= getHisto(me, verbose_,cloneME_);
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/HB Time vs Phase", process_.c_str());;
  me = mui_->get(name);
  HPHASE[0]= getHisto2(me, verbose_,cloneME_);
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/HB Time Resolution vs Energy", process_.c_str());;
  me = mui_->get(name);
  ERES[0]= getHisto2(me, verbose_,cloneME_);

  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/HO Time", process_.c_str());;
  me = mui_->get(name);
  HTIME[1]= getHisto(me, verbose_,cloneME_);
  
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/HO Time Resolution", process_.c_str());;
  me = mui_->get(name);
  HRES[1]= getHisto(me, verbose_,cloneME_);
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/HO Time vs Phase", process_.c_str());;
  me = mui_->get(name);
  HPHASE[1]= getHisto2(me, verbose_,cloneME_);
  sprintf(name,"Collector/%s/TBMonitor/TimingMonitor/HO Time Resolution vs Energy", process_.c_str());;
  me = mui_->get(name);
  ERES[1]= getHisto2(me, verbose_,cloneME_);

  return;
}

void HcalTBClient::report(){

  if ( verbose_ ) cout << "HcalTBClient: report" << endl;
  this->setup();
  
  char name[256];
  sprintf(name, "Collector/%s/TBMonitor/EventPositionMonitor/Event Position Event Number",process_.c_str());
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

void HcalTBClient::resetME(){
  
  //  MonitorElement* me;
  //  char name[150];     
  for(int i=0; i<3; i++){
    /*
    sprintf(name,"Collector/%s/HcalMonitor/TBMonitor/%s Data Format Error Words",process_.c_str(),type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);
    */
  }
  return;
}

void HcalTBClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing HcalTBClient html output ..." << endl;
  string client = "TBMonitor";
  htmlErrors(htmlDir,client,process_,mui_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Test Beam Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Test Beam</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"TBMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"TBMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"TBMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<h2><strong>Test Beam Histograms</strong></h2>" << endl;
  
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  
  this->evtposHTML(htmlDir, "TB_EvtPos.html");
  htmlFile << "<h3><a href=\"TB_EvtPos.html\">Event Position Monitor</a></h3>" << endl;
  this->timingHTML(htmlDir, "TB_Time.html");
  htmlFile << "<h3><a href=\"TB_Time.html\">Timing Monitor</a></h3>" << endl;
  this->qadcHTML(htmlDir, "TB_QADC.html");
  htmlFile << "<h3><a href=\"TB_QADC.html\">QADC Monitor</a></h3>" << endl;
  
  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();
  return;
}

void HcalTBClient::evtposHTML(string htmlDir, string htmlName){
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Event Position Monitor output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Event Position Monitor</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<h2><strong>Event Position Histograms</strong></h2>" << endl;
  
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;

  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Wire Chamber Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(WC[0],"X","Y", 92, htmlFile,htmlDir);
  histoHTML2(WC[1],"X","Y", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML2(WC[2],"X","Y", 92, htmlFile,htmlDir);
  histoHTML2(WC[3],"X","Y", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML2(WC[4],"X","Y", 92, htmlFile,htmlDir);
  histoHTML2(WC[5],"X","Y", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML2(WC[6],"X","Y", 92, htmlFile,htmlDir);
  histoHTML2(WC[7],"X","Y", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  
  histoHTML(WCX[0],"Hit Position","", 92, htmlFile,htmlDir);
  histoHTML(WCY[0],"Hit Position","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(WCX[1],"Hit Position","", 92, htmlFile,htmlDir);
  histoHTML(WCY[1],"Hit Position","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(WCX[2],"Hit Position","", 92, htmlFile,htmlDir);
  histoHTML(WCY[2],"Hit Position","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(WCX[3],"Hit Position","", 92, htmlFile,htmlDir);
  histoHTML(WCY[3],"Hit Position","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(WCX[4],"Hit Position","", 92, htmlFile,htmlDir);
  histoHTML(WCY[4],"Hit Position","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(WCX[5],"Hit Position","", 92, htmlFile,htmlDir);
  histoHTML(WCY[5],"Hit Position","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(WCX[6],"Hit Position","", 92, htmlFile,htmlDir);
  histoHTML(WCY[6],"Hit Position","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(WCX[7],"Hit Position","", 92, htmlFile,htmlDir);
  histoHTML(WCY[7],"Hit Position","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;


  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();
  return;

}

void HcalTBClient::qadcHTML(string htmlDir, string htmlName){
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: QADC Monitor output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">QADC Monitor</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<h2><strong>Cherenkov Histograms</strong></h2>" << endl;
  
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;

  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Cherenkov Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(CHK[0],"ADC","Hits/ADC", 92, htmlFile,htmlDir);
  histoHTML(CHK[1],"ADC","Hits/ADC", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(CHK[2],"ADC","Hits/ADC", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();
  return;

}

void HcalTBClient::timingHTML(string htmlDir, string htmlName){
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Timing Monitor output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Timing Monitor</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<h2><strong>TOF Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  htmlFile << "<a href=\"#Beam_Plots\">Beam Plots </a></br>" << endl;
  htmlFile << "<a href=\"#HB_Plots\">HB Plots </a></br>" << endl;
  htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;

  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\"Beam_Plots\"><h3>Timing Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(TOFT_S[0],"ADC","Hits/ADC", 92, htmlFile,htmlDir);
  histoHTML(TOFT_S[1],"ADC","Hits/ADC", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(TOFT_J[0],"ADC","Hits/ADC", 92, htmlFile,htmlDir);
  histoHTML(TOFT_J[1],"ADC","Hits/ADC", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(TOFT_S[2],"Time","Hits/nS", 92, htmlFile,htmlDir);
  histoHTML(TOFT_J[2],"Time","Hits/nS", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(TOF_DT[0],"ADC","Hits/ADC", 92, htmlFile,htmlDir);
  histoHTML(TOF_DT[1],"ADC","Hits/ADC", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(TOF_DT[2],"ADC","Hits/ADC", 100, htmlFile,htmlDir);
  //  histoHTML(DT[3],"ADC","Hits/ADC", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(TRIGT,"nS","Evts", 92, htmlFile,htmlDir);
  histoHTML(L1AT,"nS","Evts", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML(BCT,"nS","Evts", 92, htmlFile,htmlDir);
  histoHTML(PHASE,"nS","Evts", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
 htmlFile << "<hr>" << endl;
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\"HB_Plots\"><h3>HO Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(HTIME[0],"nS","Evts", 92, htmlFile,htmlDir);
  histoHTML(HRES[0],"nS","Evts", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML2(HPHASE[0],"TB Phase","Hcal Time", 92, htmlFile,htmlDir);
  histoHTML2(ERES[0],"Energy","Time", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
 htmlFile << "<hr>" << endl;
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\"HO_Plots\"><h3>HO Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(HTIME[1],"nS","Evts", 92, htmlFile,htmlDir);
  histoHTML(HRES[1],"nS","Evts", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  histoHTML2(HPHASE[1],"TB Phase","Hcal Time", 92, htmlFile,htmlDir);
  histoHTML2(ERES[1],"Energy","Time", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;


  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();
  return;

}

void HcalTBClient::createTests(){
  //  char meTitle[250], name[250];    
  //  vector<string> params;
  
  printf("There are no Test Beam quality tests yet...\n");

  return;
}

void HcalTBClient::loadHistograms(TFile* infile){
  char name[150];     
  
  TNamed* tnd = (TNamed*)infile->Get("DQMData/TBMonitor/EventPositionMonitor/Event Position Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  for(int i=0; i<3; i++){
    sprintf(name,"DQMData/TBMonitor/QADCMonitor/Cherenkov QADC %d", i+1);
     CHK[i] = (TH1F*)infile->Get(name);
  }
  for(int i=0; i<2; i++){
    sprintf(name,"DQMData/TBMonitor/QADCMonitor/TOF QADC %d", i+1);
    TOFQ[i] =(TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/TBMonitor/TimingMonitor/TOF TDC %d - Saleve", i+1);
    TOFT_S[i] =(TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/TBMonitor/TimingMonitor/TOF TDC %d - Jura", i+1);
    TOFT_J[i] =(TH1F*)infile->Get(name);
  }
  
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/TOF Time - Saleve");
  TOFT_S[2] =(TH1F*)infile->Get(name);
 
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/TOF Time - Jura");
  TOFT_J[2] =(TH1F*)infile->Get(name);

  int i = 0;
  for(char c = 'A'; c<='H'; c++){
    sprintf(name,"DQMData/TBMonitor/EventPositionMonitor/Wire Chamber %c Hits", c);
    WC[i] =(TH2F*)infile->Get(name);

    sprintf(name,"DQMData/TBMonitor/EventPositionMonitor/Wire Chamber %c X Hits", c);
    WCX[i] =(TH1F*)infile->Get(name);
 
    sprintf(name,"DQMData/TBMonitor/EventPositionMonitor/Wire Chamber %c Y Hits", c);
    WCY[i] =(TH1F*)infile->Get(name);

    i++;
  }

  sprintf(name,"DQMData/TBMonitor/TimingMonitor/TOF TDC - Delta 1");
  TOF_DT[0] = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/TBMonitor/TimingMonitor/TOF TDC - Delta 2");
  TOF_DT[1] = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/TBMonitor/TimingMonitor/TOF Time - Delta");
  TOF_DT[2] = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/TBMonitor/TimingMonitor/Trigger Timing");
  TRIGT= (TH1F*)infile->Get(name);
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/TTC L1A Timing");
  L1AT= (TH1F*)infile->Get(name);
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/Beam Coincidence Timing");
  BCT= (TH1F*)infile->Get(name);
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/TB Phase");
  PHASE= (TH1F*)infile->Get(name);
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/HB Time");
  HTIME[0]= (TH1F*)infile->Get(name);
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/HB Time Resolution");
  HRES[0]= (TH1F*)infile->Get(name);
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/HB Time vs Phase");
  HPHASE[0]= (TH2F*)infile->Get(name);
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/HB Time Resolution vs Energy");
  ERES[0]= (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/TBMonitor/TimingMonitor/HO Time");
  HTIME[1]= (TH1F*)infile->Get(name);
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/HO Time Resolution");
  HRES[1]= (TH1F*)infile->Get(name);
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/HO Time vs Phase");
  HPHASE[1]= (TH2F*)infile->Get(name);
  sprintf(name,"DQMData/TBMonitor/TimingMonitor/HO Time Resolution vs Energy");
  ERES[1]= (TH2F*)infile->Get(name);

  return;
}

void HcalTBClient::dumpHistograms(vector<TH1F*> &hist1d, vector<TH2F*> &hist2d){

  for(int i=0; i<3; i++){
    if(CHK[i]!=NULL) hist1d.push_back(CHK[i]);
    if(TOF_DT[i] !=NULL) hist1d.push_back(TOF_DT[i]);
    if(TOFT_S[i] !=NULL) hist1d.push_back(TOFT_S[i]);
    if(TOFT_J[i] !=NULL) hist1d.push_back(TOFT_J[i]);
  }

  for(int i=0; i<2; i++){
    if(TOFQ[i] !=NULL) hist1d.push_back(TOFQ[i]);
    if(HTIME[i]!=NULL) hist1d.push_back(HTIME[i]);
    if(HRES[i]!=NULL) hist1d.push_back(HRES[i]);
    if(HPHASE[i]!=NULL) hist2d.push_back(HPHASE[i]);
    if(ERES[i]!=NULL) hist2d.push_back(ERES[i]);
  }

  int i = 0;
  for(char c = 'A'; c<='H'; c++){
    if(WC[i] !=NULL) hist2d.push_back(WC[i]);
    if(WCX[i] !=NULL) hist1d.push_back(WCX[i]);
    if(WCY[i] !=NULL) hist1d.push_back(WCY[i]);
    i++;
  }

  if(TRIGT!=NULL) hist1d.push_back(TRIGT);
  if(L1AT!=NULL) hist1d.push_back(L1AT);
  if(BCT!=NULL) hist1d.push_back(BCT);
  if(PHASE!=NULL) hist1d.push_back(PHASE);

  //  printf("TBClient, names: %d, meanX: %d\n",names.size(),meanX.size());

  return;
}
