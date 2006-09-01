
#include <DQM/HcalMonitorClient/interface/HcalMonitorClient.h>

HcalMonitorClient::HcalMonitorClient(const ParameterSet& ps, MonitorUserInterface* mui){
  mui_ = mui;
  this->initialize(ps);
}

HcalMonitorClient::HcalMonitorClient(const ParameterSet& ps){
  mui_ = 0;
  this->initialize(ps);
}


HcalMonitorClient::HcalMonitorClient(){
  mui_ = 0;
  verbose_ =false;

}

HcalMonitorClient::~HcalMonitorClient(){

  cout << "HcalMonitorClient: Exit ..." << endl;

  //  if(!offline_) this->cleanup();
  this->cleanup();
  sleep(2);

  if( dataformat_client_ ) delete dataformat_client_;
  if( digi_client_ )       delete digi_client_;
  if( rechit_client_ )     delete rechit_client_;
  if( pedestal_client_ )   delete pedestal_client_;
  if( led_client_ )        delete led_client_;
  if( tb_client_ )         delete tb_client_;

  if(mui_) mui_->disconnect();
  if(mui_) delete mui_;

}

void HcalMonitorClient::initialize(const ParameterSet& ps){

  cout << endl;
  cout << " *** Hcal Generic Monitor Client ***" << endl;
  cout << endl;

  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;
  led_client_ = 0; tb_client_ = 0;

  begin_run_done_ = false;   end_run_done_   = false;
  forced_begin_run_ = false; forced_end_run_   = false;
  offline_ = false;

  status_  = "unknown"; runtype_ = "UNKNOWN";
  run_     = 0; mon_evt_     = -1;
  timeout_ = 0;

  last_jevt_   = -1; last_update_ = 0;

  // DQM default client name
  clientName_ = ps.getUntrackedParameter<string>("clientName", "HcalMonitorClient");

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "HcalMonitor");

  // DQM default collector host name
  hostName_ = ps.getUntrackedParameter<string>("hostName", "localhost");

  // DQM default host port
  hostPort_ = ps.getUntrackedParameter<int>("hostPort", 9090);

  cout << " Client '" << clientName_ << "' " << endl
       << " Collector on host '" << hostName_ << "'"
       << " on port '" << hostPort_ << "'" << endl;

  // start DQM user interface instance
  if( ! mui_ ) mui_ = new MonitorUIRoot(hostName_, hostPort_, clientName_, 5, false);

  if( verbose_ ) mui_->setVerbose(1);
  else mui_->setVerbose(0);

  // will attempt to reconnect upon connection problems (w/ a 5-sec delay)
  mui_->setReconnectDelay(5);

  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");

  if( outputFile_.size() != 0 ) {
    for ( unsigned int i = 0; i < outputFile_.size(); i++ ) {
      if( outputFile_.substr(i, 5) == ".root" )  {
        outputFile_.replace(i, 5, "");
      }
    }
  }

  // sub run enable switch
  enableSubRun_ = ps.getUntrackedParameter<bool>("enableSubRun", false);

  // location
  location_ =  ps.getUntrackedParameter<string>("location", "H2");

  // base Html output directory
  baseHtmlDir_ = ps.getUntrackedParameter<string>("baseHtmlDir", "");

  if( baseHtmlDir_.size() != 0 ) cout << " HTML output will go to"
				       << " baseHtmlDir = '" << baseHtmlDir_ << "'" << endl;
  else cout << " HTML output is disabled" << endl;
  
  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  if( cloneME_ ) cout << " cloneME switch is ON" << endl;
  else cout << " cloneME switch is OFF" << endl;

  // exit on end job switch
  enableExit_ = ps.getUntrackedParameter<bool>("enableExit", true);

  if( enableExit_ ) cout << " enableExit switch is ON" << endl;
  else cout << " enableExit switch is OFF" << endl;

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if( verbose_ ) cout << " verbose switch is ON" << endl;
  else cout << " verbose switch is OFF" << endl;

  update_freq_ = ps.getUntrackedParameter<int>("updateFrequency", 1000);
  timeout_thresh_ = ps.getUntrackedParameter<int>("Timeout", 100);

  // global ROOT style
  gStyle->Reset("Default");
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetFillColor(0);
  gStyle->SetTitleFillColor(10);
  //  gStyle->SetOptStat(0);


  // clients' constructors
  if( ps.getUntrackedParameter<bool>("DataFormatClient", false) )
    dataformat_client_   = new HcalDataFormatClient(ps, mui_);
  if( ps.getUntrackedParameter<bool>("DigiClient", false) )
    digi_client_         = new HcalDigiClient(ps, mui_);
  if( ps.getUntrackedParameter<bool>("RecHitClient", false) )
    rechit_client_       = new HcalRecHitClient(ps, mui_);
  if( ps.getUntrackedParameter<bool>("PedestalClient", false) )
    pedestal_client_     = new HcalPedestalClient(ps, mui_);
  if( ps.getUntrackedParameter<bool>("LEDClient", false) )
    led_client_          = new HcalLEDClient(ps, mui_);
  if( ps.getUntrackedParameter<bool>("TestBeamClient", false) )
    tb_client_          = new HcalTBClient(ps, mui_);

  return;
}

void HcalMonitorClient::beginJob(const EventSetup& eventSetup){

  if( verbose_ ) cout << "HcalMonitorClient: beginJob" << endl;
  
  ievt_ = 0;
  jevt_ = 0;

  this->subscribe();

  if( dataformat_client_ ) dataformat_client_->beginJob();
  if( digi_client_ )       digi_client_->beginJob();
  if( rechit_client_ )     rechit_client_->beginJob();
  if( pedestal_client_ )   pedestal_client_->beginJob(eventSetup);
  if( led_client_ )        led_client_->beginJob(eventSetup);
  if( tb_client_ )         tb_client_->beginJob();
    
  return;
}

void HcalMonitorClient::beginRun(void){

  if( verbose_ ) cout << "HcalMonitorClient: beginRun" << endl;

  this->setup();

  if( dataformat_client_ ) dataformat_client_->beginRun();
  if( digi_client_ )       digi_client_->beginRun();
  if( rechit_client_ )     rechit_client_->beginRun();
  if( pedestal_client_ )   pedestal_client_->beginRun();
  if( led_client_ )        led_client_->beginRun();
  if( tb_client_ )         tb_client_->beginRun();

  return;

}

void HcalMonitorClient::endJob(void) {

  if( verbose_ ) cout << "HcalMonitorClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

  if( tb_client_ ) tb_client_->endJob();
  if( dataformat_client_ ) dataformat_client_->endJob();
  if( digi_client_ )  digi_client_->endJob();
  if( rechit_client_ )  rechit_client_->endJob();
  if( pedestal_client_ ) pedestal_client_->endJob();
  if( led_client_ ) led_client_->endJob();

  return;
}

void HcalMonitorClient::report(bool doUpdate) {
  
  if( verbose_ ) cout << "HcalMonitorClient: creating report, ievt = " << ievt_ << endl;
  
  if(doUpdate && status_!="unknown"){
    this->createTests();  
    mui_->update();
  }
  //  if(doUpdate) mui_->runQTests();
  
  if( outputFile_.size() != 0) {    
    for( unsigned int i = 0; i < outputFile_.size(); i++ ) {
      if( outputFile_.substr(i, 5) == ".root" )  {
        outputFile_.replace(i, 5, "");
      }
    }
    char tmp[150];
    sprintf(tmp,"%09d.root", run_);
    string saver = outputFile_+tmp;
    mui_->save(saver);
    
    TFile* rootOut = new TFile(saver.c_str(),"UPDATE");
    rootOut->cd();
    map<string, vector<QReport*> > errE, errW, errO;
    
    if( tb_client_ ) {      
      tb_client_->report();
      tb_client_->getErrors(errE,errW,errO);
    }    
    if( led_client_ ) {      
      led_client_->report();
      led_client_->getErrors(errE,errW,errO);
    }
    if( pedestal_client_ ) {      
      pedestal_client_->report();
      pedestal_client_->getErrors(errE,errW,errO);
    }
    if( digi_client_ ) {
      digi_client_->report();
      digi_client_->getErrors(errE,errW,errO);
    }
    if( rechit_client_ ) {
      rechit_client_->report();
      rechit_client_->getErrors(errE,errW,errO);
    }    
    if( dataformat_client_ ) {
      dataformat_client_->report();
      dataformat_client_->getErrors(errE,errW,errO);
    }

    rootOut->Write();
    rootOut->Close();
  }

  if( baseHtmlDir_.size() != 0 ) this->htmlOutput();

  status_="unknown";

  return;
}


void HcalMonitorClient::endRun(void) {
  if( verbose_ ) cout << "HcalMonitorClient: endRun, jevt = " << jevt_ << endl;
 
  this->report(false);
  
  if( tb_client_ )         tb_client_->endRun();
  if( dataformat_client_ ) dataformat_client_->endRun();
  if( digi_client_ )       digi_client_->endRun();
  if( rechit_client_ )     rechit_client_->endRun();
  if( pedestal_client_ )   pedestal_client_->endRun();
  if( led_client_ )        led_client_->endRun();
  
  this->cleanup();
  
  status_  = "unknown";
  mon_evt_     = -1;
  runtype_ = "UNKNOWN";
  
  last_jevt_ = -1;
  last_update_ = 0;
  run_=0;
  
  // this is an effective way to avoid ROOT memory leaks ...
  if( enableExit_ ) {
    cout << endl;
    cout << ">>> exit after End-Of-Run <<<" << endl;
    cout << endl;
    this->endJob();
    throw exception();
  }

  return;
}

void HcalMonitorClient::setup(void) {
  return;
}

void HcalMonitorClient::cleanup(void) {  
  this->unsubscribe();
  return;
}

void HcalMonitorClient::subscribe(void){

  if( verbose_ ) cout << "HcalMonitorClient: subscribe" << endl;

  // subscribe to monitorable matching pattern
  mui_->subscribe("*/HcalMonitor/STATUS");
  mui_->subscribe("*/HcalMonitor/RUN NUMBER");
  mui_->subscribe("*/HcalMonitor/EVT NUMBER");
  mui_->subscribe("*/HcalMonitor/EVT MASK");
  mui_->subscribe("*/HcalMonitor/RUN TYPE");

  if( dataformat_client_ ) dataformat_client_->subscribe();
  if( digi_client_ ) digi_client_->subscribe();
  if( rechit_client_ ) rechit_client_->subscribe();
  if( pedestal_client_ ) pedestal_client_->subscribe();  
  if( led_client_ ) led_client_->subscribe();
  if( tb_client_ ) tb_client_->subscribe();

  return;
}

void HcalMonitorClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/HcalMonitor/STATUS");
  mui_->subscribeNew("*/HcalMonitor/RUN NUMBER");
  mui_->subscribeNew("*/HcalMonitor/EVT NUMBER");
  mui_->subscribeNew("*/HcalMonitor/EVT MASK");
  mui_->subscribeNew("*/HcalMonitor/RUN TYPE");

  if( dataformat_client_ ) dataformat_client_->subscribeNew();
  if( digi_client_ ) digi_client_->subscribeNew();
  if( rechit_client_ ) rechit_client_->subscribeNew();
  if( pedestal_client_ ) pedestal_client_->subscribeNew();  
  if( led_client_ ) led_client_->subscribeNew();
  if( tb_client_ ) tb_client_->subscribeNew();

  return;
}

void HcalMonitorClient::unsubscribe(void) {

  if( verbose_ ) cout << "HcalMonitorClient: unsubscribe" << endl;

  // unsubscribe to all monitorable matching pattern
  if(mui_){
    mui_->unsubscribe("*/HcalMonitor/STATUS");
    mui_->unsubscribe("*/HcalMonitor/RUN NUMBER");
    mui_->unsubscribe("*/HcalMonitor/EVT NUMBER");
    mui_->unsubscribe("*/HcalMonitor/EVT MASK");
    mui_->unsubscribe("*/HcalMonitor/RUN TYPE");
  }

  if( tb_client_ ) tb_client_->unsubscribe();
  if( dataformat_client_ ) dataformat_client_->unsubscribe();
  if( digi_client_ ) digi_client_->unsubscribe();
  if( rechit_client_ ) rechit_client_->unsubscribe();
  if( pedestal_client_ ) pedestal_client_->unsubscribe();  
  if( led_client_ ) led_client_->unsubscribe();
  
  return;
}

void HcalMonitorClient::analyze(const Event& e, const edm::EventSetup& eventSetup){
  
  ievt_++;
  
  printf("Client heartbeat....\n");
  
  Char_t histo[150];
  MonitorElement* me;
  string s;  
  
  // # of full monitoring cycles processed
  int updates = mui_->getNumUpdates();
  
  //  mui_->update();
  mui_->doMonitoring();
  this->subscribeNew();

  bool force_update = false;
  if(timeout_>=timeout_thresh_){
    printf("\n\n\n\nHcalMonitorClient: Forcing update after timeout!\n\n\n\n");
    force_update = true;
    status_ = "end-of-run";
  }
  
  if( updates != last_update_ || force_update) {
    int lastRun = run_;  
    sprintf(histo, "Collector/%s/HcalMonitor/RUN NUMBER",process_.c_str());
    me = mui_->get(histo);
    if( me ) {
      s = me->valueString();
      run_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &run_);
      if( verbose_ ) cout << "Found '" << histo << "'" << endl;
    }
    
    sprintf(histo, "Collector/%s/HcalMonitor/EVT NUMBER",process_.c_str());
    me = mui_->get(histo);
    if( me ) {
      s = me->valueString();
      mon_evt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &mon_evt_);
      if( verbose_ ) cout << "Found '" << histo << "'" << endl;
    }
    
    sprintf(histo, "Collector/%s/HcalMonitor/EVT MASK",process_.c_str());
    me = mui_->get(histo);
    if( me ) {
      s = me->valueString();
      int mask = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &mask);
      if( verbose_ ) cout << "Found '" << histo << "'" << endl;
      if(mask&HCAL_BEAM_TRIGGER) runtype_ = "BEAM RUN";
      if(mask&DO_HCAL_PED_CALIBMON){ runtype_ = "PEDESTAL RUN";
      if(mask&HCAL_BEAM_TRIGGER) runtype_ = "BEAM AND PEDESTALS";
      }
      if(mask&DO_HCAL_LED_CALIBMON) runtype_ = "LED RUN";
      if(mask&DO_HCAL_LASER_CALIBMON) runtype_ = "LASER RUN";
    }
    
    status_="unknown";
    sprintf(histo, "Collector/%s/HcalMonitor/STATUS",process_.c_str());
    me = mui_->get(histo);
    if( me ) {
      s = me->valueString();
      status_ = "unknown";
      if( s.substr(2,1) == "0" ) status_ = "begin-of-run";
      if( s.substr(2,1) == "1" ) status_ = "running";
      if( s.substr(2,1) == "2" ) status_ = "end-of-run";
      if( verbose_ ) cout << "Found '" << histo << "'" << endl;
    }
    if( verbose_ ) printf("Status: %s\n",status_.c_str());  
    printf("Status: %s\n",status_.c_str());
        
    if(status_=="begin-of-run") this->beginRun();
    if(status_=="running"){
      if( dataformat_client_ ) dataformat_client_->analyze(); 	
      if( digi_client_ )       digi_client_->analyze(); 
      if( rechit_client_ )     rechit_client_->analyze(); 
      if( pedestal_client_ )   pedestal_client_->analyze();      
      if( led_client_ )        led_client_->analyze(); 
      if( tb_client_ )         tb_client_->analyze(); 
    }    
    if(status_ == "end-of-run") this->endRun();
    if(status_!="unknown") 
      if(run_!=lastRun && lastRun!=0 && mon_evt_>1) this->report(false);    
    if((ievt_%update_freq_)==0 && mon_evt_>1 && status_=="running") this->report(true);
    if(status_!="unknown" && (ievt_%10)==0 ){
      if((ievt_%update_freq_)!=0 || status_ == "begin-of-run"){ 
	createTests();  
	mui_->update();
      }
    }
    
    timeout_=0;
  }
  else timeout_++;
  
  last_update_ = updates;
  
  return;
}

void HcalMonitorClient::createTests(void){

  if( dataformat_client_ ) dataformat_client_->createTests(); 
  if( digi_client_ ) digi_client_->createTests(); 
  if( rechit_client_ ) rechit_client_->createTests(); 
  if( pedestal_client_ ) pedestal_client_->createTests(); 
  if( led_client_ ) led_client_->createTests(); 
  if( tb_client_ ) tb_client_->createTests(); 

  return;
}

void HcalMonitorClient::htmlOutput(void){

  cout << "Preparing HcalMonitorClient html output ..." << endl;

  char tmp[10];
  sprintf(tmp, "%09d", run_);

  string htmlDir = baseHtmlDir_ + "/" + tmp + "/";
  system(("/bin/mkdir -p " + htmlDir).c_str());

  ofstream htmlFile;
  htmlFile.open((htmlDir + "index.html").c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Hcal Data Quality Monitor</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<center><h1>Hcal Data Quality Monitor</h1></center>" << endl;
  htmlFile << "<h2>Run Number:&nbsp&nbsp&nbsp" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << run_ <<"</span></h2> " << endl;
  htmlFile << "<h2>Run type:&nbsp&nbsp&nbsp" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << runtype_ <<"</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp&nbsp&nbsp" << endl;
  htmlFile << "<span style=\"color: rgb(0, 0, 153);\">" << mon_evt_ <<"</span></h2> " << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<ul>" << endl;

  string htmlName;

  // Dataformat check
  if( tb_client_ ) {
    htmlName = "TestBeamClient.html";
    tb_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Test Beam Monitor</a></td" << endl;
    if(tb_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(tb_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(tb_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</table>" << endl;
  }
  if( dataformat_client_ ) {
    htmlName = "HcalDataFormatClient.html";
    dataformat_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Data Format Monitor</a></td" << endl;
    if(dataformat_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(dataformat_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(dataformat_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</table>" << endl;
  }
  if( digi_client_ ) {
    htmlName = "HcalDigiClient.html";
    digi_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Digi Monitor</a></td" << endl;
    if(digi_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(digi_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(digi_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</table>" << endl;
  }
  if( rechit_client_ ) {
    htmlName = "HcalRecHitClient.html";
    rechit_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">RecHit Monitor</a></td" << endl;
    if(rechit_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(rechit_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(rechit_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    htmlFile << "</table>" << endl;
  }

  if( pedestal_client_) {
    htmlName = "HcalPedestalClient.html";
    pedestal_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Pedestal Monitor</a></td" << endl;
    
    if(pedestal_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(pedestal_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(pedestal_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</table>" << endl;
  }

  if( led_client_) {
    htmlName = "HcalLEDClient.html";
    led_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">LED Monitor</a></td" << endl;
    
    if(led_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(led_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(led_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
    else htmlFile << "<td bgcolor=lime align=center>This monitor task has no problems</td>" << endl;
    
    htmlFile << "</table>" << endl;
  }

  htmlFile << "</ul>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  cout << endl;

}
void HcalMonitorClient::offlineSetup(){
  cout << endl;
  cout << " *** Hcal Generic Monitor Client, for offline operation***" << endl;
  cout << endl;

  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;
  led_client_ = 0; tb_client_ = 0;

  begin_run_done_ = false;   end_run_done_   = false;
  forced_begin_run_ = false; forced_end_run_   = false;
  offline_ = true;

  status_  = "unknown"; runtype_ = "UNKNOWN";
  run_     = 0; mon_evt_     = -1;
  timeout_ = 0;

  last_jevt_   = -1; last_update_ = 0;

  // base Html output directory
  baseHtmlDir_ = ".";
  
  // clients' constructors
  tb_client_           = new HcalTBClient();
  dataformat_client_   = new HcalDataFormatClient();
  rechit_client_       = new HcalRecHitClient();
  digi_client_         = new HcalDigiClient();
  pedestal_client_     = new HcalPedestalClient();
  led_client_          = new HcalLEDClient();

  return;
}

void HcalMonitorClient::loadHistograms(TFile* infile){
  
  if(!infile){
    throw cms::Exception("Incomplete configuration") << 
      "HcalMonitorClient: this histogram file is bad! " <<endl;
    return;
  }
    
  TNamed* tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/RUN NUMBER");
  string s =tnd->GetTitle();
  run_ = -1;
  sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &run_);
  
  tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/EVT NUMBER");
  s =tnd->GetTitle();
  mon_evt_ = -1;
  sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &mon_evt_);

  tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/EVT MASK");
  s =tnd->GetTitle();
  int mask = -1;
  sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &mask);
  
  if(mask&HCAL_BEAM_TRIGGER) runtype_ = "BEAM RUN";
  if(mask&DO_HCAL_PED_CALIBMON){ runtype_ = "PEDESTAL RUN";
    if(mask&HCAL_BEAM_TRIGGER) runtype_ = "BEAM AND PEDESTALS";
  }
  if(mask&DO_HCAL_LED_CALIBMON) runtype_ = "LED RUN";
  if(mask&DO_HCAL_LASER_CALIBMON) runtype_ = "LASER RUN";
  
  status_="unknown";
  tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/EVT MASK");
  s = tnd->GetTitle();
  status_ = "unknown";
  if( s.substr(2,1) == "0" ) status_ = "begin-of-run";
  if( s.substr(2,1) == "1" ) status_ = "running";
  if( s.substr(2,1) == "2" ) status_ = "end-of-run";
  
  printf("HcalOfflineClient: run: %d, evts: %d, type: %s, status: %s\n",run_, mon_evt_, runtype_.c_str(),status_.c_str());


  if(tb_client_) tb_client_->loadHistograms(infile);
  if(dataformat_client_) dataformat_client_->loadHistograms(infile);
  if(rechit_client_) rechit_client_->loadHistograms(infile);
  if(digi_client_) digi_client_->loadHistograms(infile);
  if(pedestal_client_) pedestal_client_->loadHistograms(infile);
  if(led_client_) led_client_->loadHistograms(infile);

  return;

}


