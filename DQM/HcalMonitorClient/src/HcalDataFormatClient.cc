#include <DQM/HcalMonitorClient/interface/HcalDataFormatClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>

HcalDataFormatClient::HcalDataFormatClient(const ParameterSet& ps, MonitorUserInterface* mui){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = mui;
  for(int i=0; i<3; i++){
    dferr_[i] = NULL;
    //    crateErrMap_[i] =NULL;
    //    spigotErrMap_[i] = NULL;
  }
  spigotErrs_ = NULL;
  badDigis_ = NULL;
  unmappedDigis_ = NULL;
  unmappedTPDs_ = NULL;
  fedErrMap_ = NULL;
  BCN_ = NULL;

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

  ievt_=0; jevt_=0;

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "HcalMonitor");

  vector<string> subdets = ps.getUntrackedParameter<vector<string> >("subDetsOn");
  for(int i=0; i<3; i++) subDetsOn_[i] = false;
  
  for(unsigned int i=0; i<subdets.size(); i++){
    if(subdets[i]=="HBHE") subDetsOn_[0] = true;
    else if(subdets[i]=="HF") subDetsOn_[1] = true;
    else if(subdets[i]=="HO") subDetsOn_[2] = true;
  }
}

HcalDataFormatClient::HcalDataFormatClient(){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  verbose_ =false;
  mui_ = 0;
  for(int i=0; i<3; i++){
    dferr_[i] = NULL;
    //   crateErrMap_[i] =NULL;
    //    spigotErrMap_[i] = NULL;
  }

  spigotErrs_ = NULL;
  badDigis_ = NULL;
  unmappedDigis_ = NULL;
  unmappedTPDs_ = NULL;
  fedErrMap_ = NULL;
  BCN_ = NULL;
 
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

  for(int i=0; i<3; i++) subDetsOn_[i] = false;
}

HcalDataFormatClient::~HcalDataFormatClient(){

  this->cleanup();
  
}

void HcalDataFormatClient::beginJob(void){

  if ( verbose_ ) cout << "HcalDataFormatClient: beginJob" << endl;
  ievt_ = 0; jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetAllME();
  return;
}

void HcalDataFormatClient::beginRun(void){

  if ( verbose_ ) cout << "HcalDataFormatClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetAllME();
  return;
}

void HcalDataFormatClient::endJob(void) {

  if ( verbose_ ) cout << "HcalDataFormatClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();
  return;
}

void HcalDataFormatClient::endRun(void) {

  if ( verbose_ ) cout << "HcalDataFormatClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();
  return;
}

void HcalDataFormatClient::setup(void) {
  return;
}

void HcalDataFormatClient::cleanup(void) {

  if ( cloneME_ ) {
    for(int i=0; i<3; i++){
      if ( dferr_[i] ) delete dferr_[i];
      //      if ( crateErrMap_[i]) delete crateErrMap_[i];
      //      if ( spigotErrMap_[i]) delete spigotErrMap_[i];
    }
  
    if ( spigotErrs_) delete spigotErrs_;
    if ( badDigis_) delete badDigis_;
    if ( unmappedDigis_) delete unmappedDigis_;
    if ( unmappedTPDs_) delete unmappedTPDs_;
    if ( fedErrMap_) delete fedErrMap_;

    if( BCN_) delete BCN_;

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

  }  
  for(int i=0; i<3; i++){
    dferr_[i] = NULL;
    //    crateErrMap_[i] =NULL;
    //    spigotErrMap_[i] = NULL;
  }
  
  spigotErrs_ = NULL;
  badDigis_ = NULL;
  unmappedDigis_ = NULL;
  unmappedTPDs_ = NULL;
  fedErrMap_ = NULL;


  BCN_ = NULL;

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
    MonitorElement* me = mui_->getBEInterface()->get(meName);
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
  if(!mui_) return;
  
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

  sprintf(name,"DataFormatMonitor/BCN");
  BCN_ = getHisto(name, process_, mui_, verbose_,cloneME_);

  sprintf(name,"DataFormatMonitor/Evt Number Out-of-Synch");
  EvtMap_ = getHisto2(name, process_, mui_, verbose_,cloneME_);

  sprintf(name,"DataFormatMonitor/BCN Not Constant");
  BCNMap_ = getHisto2(name, process_, mui_, verbose_,cloneME_);

  sprintf(name,"DataFormatMonitor/HTR Firmware Version");
  FWVerbyCrate_ = getHisto2(name, process_, mui_, verbose_,cloneME_);

  sprintf(name,"DataFormatMonitor/HTR Error Word by Crate");
  ErrMapbyCrate_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrMapbyCrate_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 0");
  ErrCrate0_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate0_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 1");
  ErrCrate1_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate1_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 2");
  ErrCrate2_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate2_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 3");
  ErrCrate3_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate3_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 4");
  ErrCrate4_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate4_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 5");
  ErrCrate5_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate5_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 6");
  ErrCrate6_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate6_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 7");
  ErrCrate7_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate7_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 8");
  ErrCrate8_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate8_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 9");
  ErrCrate9_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate9_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 10");
  ErrCrate10_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate10_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 11");
  ErrCrate11_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate11_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 12");
  ErrCrate12_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate12_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 13");
  ErrCrate13_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate13_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 14");
  ErrCrate14_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate14_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 15");
  ErrCrate15_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate15_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 16");
  ErrCrate16_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate16_);

  sprintf(name,"DataFormatMonitor/HTR Error Word - Crate 17");
  ErrCrate17_ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  labelyBits(ErrCrate17_);
 
  for(int i=0; i<3; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HBHE";
    if(i==1) type = "HF";
    else if(i==2) type = "HO";
    sprintf(name,"DataFormatMonitor/%s Data Format Error Word", type.c_str());
    dferr_[i] = getHisto(name, process_, mui_, verbose_,cloneME_);    
    labelxBits(dferr_[i]);
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
  if(!mui_) return;
  if ( verbose_ ) cout << "HcalDataFormatClient: report" << endl;
  this->setup();
  
  char name[256];
  sprintf(name, "%sHcalMonitor/DataFormatMonitor/Data Format Task Event Number",process_.c_str());
  MonitorElement* me = mui_->getBEInterface()->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( verbose_ ) cout << "Found '" << name << "'" << endl;
  }

  getHistograms();
  
  return;
}

void HcalDataFormatClient::resetAllME(){

  if(!mui_) return;
  
  char name[150];     
  sprintf(name,"%sHcalMonitor/DataFormatMonitor/Spigot Format Errors",process_.c_str());
  resetME(name,mui_);
  
  sprintf(name,"%sHcalMonitor/DataFormatMonitor/Bad Quality Digis",process_.c_str());
  resetME(name,mui_);
  
  sprintf(name,"%sHcalMonitor/DataFormatMonitor/Unmapped Digis",process_.c_str());
  resetME(name,mui_);
  
  sprintf(name,"%sHcalMonitor/DataFormatMonitor/Unmapped Trigger Primitive Digis",process_.c_str());
  resetME(name,mui_);
  
  sprintf(name,"%sHcalMonitor/DataFormatMonitor/FED Error Map",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/BCN",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/Evt Number Out-of-Synch",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/BCN Not Constant",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Firmware Version",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word by Crate",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 0",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 1",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 2",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 3",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 4",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 5",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 6",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 7",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 8",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 9",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 10",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 11",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 12",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 13",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 14",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 15",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 16",process_.c_str());
  resetME(name,mui_);

  sprintf(name,"%sHcalMonitor/DataFormatMonitor/HTR Error Word - Crate 17",process_.c_str());
  resetME(name,mui_);

  for(int i=0; i<3; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HBHE";
    if(i==1) type = "HF";
    else if(i==2) type = "HO";

    sprintf(name,"%sHcalMonitor/DataFormatMonitor/%s Data Format Error Word",process_.c_str(), type.c_str());
    resetME(name,mui_);
    /*
    sprintf(name,"%sHcalMonitor/DataFormatMonitor/%s Data Format Crate Error Map",process_.c_str(), type.c_str());
    resetME(name,mui_);

    sprintf(name,"%sHcalMonitor/DataFormatMonitor/%s Data Format Spigot Error Map",process_.c_str(), type.c_str());
    resetME(name,mui_);
    */   
  }
  
  return;
}

void HcalDataFormatClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  cout << "Preparing HcalDataFormatClient html output ..." << endl;
  string client = "DataFormatMonitor";
  htmlErrors(runNo,htmlDir,client,process_,mui_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);

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
  if(subDetsOn_[0]) htmlFile << "<a href=\"#HBHE_Plots\">HBHE Plots </a></br>" << endl;
  if(subDetsOn_[1]) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  if(subDetsOn_[2]) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;

  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,ErrMapbyCrate_,"Crate #"," ", 92, htmlFile,htmlDir);
  histoHTML(runNo,BCN_,"Bunch Counter Number","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,BCNMap_,"Slot #","Crate #", 92, htmlFile,htmlDir);
  histoHTML2(runNo,EvtMap_,"Slot #","Crate #", 100, htmlFile,htmlDir);
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
  histoHTML(runNo,fedErrMap_,"DCC id","# Errors", 92, htmlFile,htmlDir);
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



  for(int i=0; i<3; i++){
    if(!subDetsOn_[i]) continue;
    
    string type = "HBHE";
    if(i==1) type = "HF"; 
    else if(i==2) type = "HO"; 
    
    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;
    htmlFile << "<tr align=\"left\">" << endl;  
    histoHTML(runNo,dferr_[i],"Error Bit","Frequency", 92, htmlFile,htmlDir);
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
  if(!mui_) return;

  char meTitle[250], name[250];    
  vector<string> params;
  
  if(verbose_) printf("Creating Data Format tests...\n"); 
  
  for(int i=0; i<3; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HBHE";
    if(i==1) type = "HF"; 
    else if(i==2) type = "HO"; 
    
    sprintf(meTitle,"%sHcalMonitor/DataFormatMonitor/%s Data Format Error Word",process_.c_str(),type.c_str());
    sprintf(name,"%s DataFormat",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = mui_->getBEInterface()->get(meTitle);
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

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/BCN");
  BCN_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/Evt Number Out-of-Synch");
  EvtMap_ = (TH2F*)infile->Get(name);
  
  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/BCN Not Constant");
  BCNMap_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Firmware Version");
  FWVerbyCrate_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word by Crate");
  ErrMapbyCrate_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 0");
  ErrCrate0_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 1");
  ErrCrate1_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 2");
  ErrCrate2_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 3");
  ErrCrate3_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 4");
  ErrCrate4_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 5");
  ErrCrate5_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 6");
  ErrCrate6_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 7");
  ErrCrate7_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 8");
  ErrCrate8_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 9");
  ErrCrate9_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 10");
  ErrCrate10_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 11");
  ErrCrate11_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 12");
  ErrCrate12_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 13");
  ErrCrate13_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 14");
  ErrCrate14_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 15");
  ErrCrate15_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 16");
  ErrCrate16_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/HTR Error Word - Crate 17");
  ErrCrate17_ = (TH2F*)infile->Get(name);

  for(int i=0; i<3; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HBHE";
    if(i==1) type = "HF";
    else if(i==2) type = "HO";

    sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/%s Data Format Error Word", type.c_str());
    dferr_[i] = (TH1F*)infile->Get(name);    
    labelxBits(dferr_[i]);
    /*    
    sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/%s Data Format Crate Error Map", type.c_str());
    crateErrMap_[i] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/DataFormatMonitor/%s Data Format Spigot Error Map", type.c_str());
    spigotErrMap_[i] = (TH2F*)infile->Get(name);
    */
  }

  return;
}
