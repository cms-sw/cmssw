#include <DQM/HcalMonitorClient/interface/HcalMonitorClient.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

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
}

HcalMonitorClient::~HcalMonitorClient(){

  cout << "HcalMonitorClient: Exit ..." << endl;
  
  this->cleanup();

  if( dataformat_client_ ) delete dataformat_client_;
  if( digi_client_ )       delete digi_client_;
  if( rechit_client_ )     delete rechit_client_;
  if( pedestal_client_ )   delete pedestal_client_;
  if( led_client_ )        delete led_client_;
  if( hot_client_ )         delete hot_client_;

  if(mui_) mui_->disconnect();
}

void HcalMonitorClient::initialize(const ParameterSet& ps){

  cout << endl;
  cout << " *** Hcal Generic Monitor Client ***" << endl;
  cout << endl;

  subscribed_=0;

  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;
  led_client_ = 0;
  hot_client_ = 0;
  trigger_=0;
  
  offline_ = false; verbose_=false;

  status_  = "unknown"; runtype_ = "UNKNOWN";
  run_     = -1; mon_evt_     = -1;
  nTimeouts_ = 0;
  
  last_mon_evt_   = -1; 
  last_update_ = 0;
  last_reset_Evts_=0;

  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if( outputFile_.size() != 0 ) {
    for ( unsigned int i = 0; i < outputFile_.size(); i++ ) {
      if( outputFile_.substr(i, 5) == ".root" )  {
        outputFile_.replace(i, 5, "");
      }
    }
  }

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);
  
  if ( enableMonitorDaemon_ ) cout << "-->enableMonitorDaemon switch is ON" << endl;
  else cout << "-->enableMonitorDaemon switch is OFF" << endl;

  // DQM ROOT input
  inputFile_ = ps.getUntrackedParameter<string>("inputFile", "");
  if(inputFile_.size()!=0 ) cout << "-->reading DQM input from " << inputFile_ << endl;

  // DQM default client name
  clientName_ = ps.getUntrackedParameter<string>("clientName", "HcalMonitorClient");

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "");

  // DQM default collector host name
  hostName_ = ps.getUntrackedParameter<string>("hostName", "localhost");

  // DQM default host port
  hostPort_ = ps.getUntrackedParameter<int>("hostPort", 9090);

  cout << "-->Client '" << clientName_ << "' " << endl
       << "-->Collector on host '" << hostName_ << "'"
       << " on port '" << hostPort_ << "'" << endl;


  // Server switch
  enableServer_ = ps.getUntrackedParameter<bool>("enableServer", false);
  serverPort_   = ps.getUntrackedParameter<int>("serverPort", 9900);
  
  if( enableServer_ ) {
    cout << "-->enableServer switch is ON" << endl;
    if( enableMonitorDaemon_ && hostPort_ != serverPort_ ) {
      cout << "-->Forcing the same port for Collector and Server" << endl;
      serverPort_ = hostPort_;
    }
    cout << "-->Running server on port '" << serverPort_ << "'" << endl;
  } else cout << "-->enableServer switch is OFF" << endl;


  // location
  location_ =  ps.getUntrackedParameter<string>("location", "USC");

  //histogram reset freqency, update frequency, timeout
  resetUpdate_ = ps.getUntrackedParameter<int>("resetFreqUpdates",-1);  //number of collector updates
  if(resetUpdate_!=-1) cout << "-->Will reset histograms every " << resetUpdate_ <<" collector updates." << endl;
  resetEvents_ = ps.getUntrackedParameter<int>("resetFreqEvents",-1);   //number of real events
  if(resetEvents_!=-1) cout << "-->Will reset histograms every " << resetEvents_ <<" events." << endl;
  resetTime_ = ps.getUntrackedParameter<int>("resetFreqTime",-1);       //number of minutes
  if(resetTime_!=-1) cout << "-->Will reset histograms every " << resetTime_ <<" minutes." << endl;

  nUpdateEvents_ = ps.getUntrackedParameter<int>("outputFreqEvts", 1000);
  cout << "-->Will produce output every " << nUpdateEvents_ <<" events." << endl;
  timeoutThresh_ = ps.getUntrackedParameter<int>("Timeout", 100);
  cout << "-->Timeout threshold set to " << timeoutThresh_ <<" cycles." << endl;

  // base Html output directory
  baseHtmlDir_ = ps.getUntrackedParameter<string>("baseHtmlDir", "");

  if( baseHtmlDir_.size() != 0 ) 
    cout << "-->HTML output will go to baseHtmlDir = '" << baseHtmlDir_ << "'" << endl;
  else cout << "-->HTML output is disabled" << endl;
  
  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  if( cloneME_ ) cout << "-->cloneME switch is ON" << endl;
  else cout << "-->cloneME switch is OFF" << endl;

  // exit on end job switch
  enableExit_ = ps.getUntrackedParameter<bool>("enableExit", true);

  if( enableExit_ ) cout << "-->enableExit switch is ON" << endl;
  else cout << "-->enableExit switch is OFF" << endl;

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if( verbose_ ) cout << "-->verbose switch is ON" << endl;
  else cout << "-->verbose switch is OFF" << endl;

  // mergeRuns switch

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  if ( mergeRuns_ ) cout << " mergeRuns switch is ON" << endl;
  else cout << " mergeRuns switch is OFF" << endl;

  // global ROOT style
  gStyle->Reset("Default");
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetFillColor(0);
  gStyle->SetTitleFillColor(10);
  //  gStyle->SetOptStat(0);
  gStyle->SetPalette(1);


  //mui_->getBEInterface() = edm::Service<DaqMonitorBEInterface>().operator->();
  
  //if ( verbose_ ) mui_->getBEInterface()->setVerbose(1);
  //else mui_->getBEInterface()->setVerbose(0);
  
  if ( enableMonitorDaemon_ ) {
    if ( enableServer_ ) {
      mui_ = new MonitorUIRoot(hostName_, hostPort_, clientName_, 5, true);
    } else {
      mui_ = new MonitorUIRoot(hostName_, hostPort_, clientName_, 5, false);
    }
  } else {
    mui_ = new MonitorUIRoot();
    if ( enableServer_ ) {
      mui_->actAsServer(serverPort_, clientName_);
    }
  }
  
  if ( verbose_ ) mui_->getBEInterface()->setVerbose(1);
  else mui_->getBEInterface()->setVerbose(0);
  
  if( ! enableMonitorDaemon_ ) {  
    if( inputFile_.size() != 0 && mui_->getBEInterface()!=NULL){
      mui_->getBEInterface()->open(inputFile_);
      mui_->getBEInterface()->showDirStructure();     
    }
  }

  if(mui_) mui_->setMaxAttempts2Reconnect(99999);




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
  if( ps.getUntrackedParameter<bool>("HotCellClient", false) )
    hot_client_          = new HcalHotCellClient(ps, mui_);


  gettimeofday(&startTime_,NULL);

  return;
}
// remove all MonitorElements and directories
void HcalMonitorClient::removeAll(){

  if(mui_->getBEInterface()==NULL) return;

  mui_->getBEInterface()->setVerbose(0);


  // go to top directory
  mui_->getBEInterface()->cd();
  // remove MEs at top directory
  mui_->getBEInterface()->removeContents(); 
  // remove directory (including subdirectories recursively)
  if(mui_->getBEInterface()->dirExists("Collector"))
     mui_->getBEInterface()->rmdir("Collector");
  if(mui_->getBEInterface()->dirExists("Summary"))
  mui_->getBEInterface()->rmdir("Summary");

  mui_->getBEInterface()->setVerbose(1);
}

void HcalMonitorClient::beginJob(const EventSetup& eventSetup){
  
  if( verbose_ ) cout << "HcalMonitorClient: beginJob" << endl;
  
  ievt_ = 0;
  last_run_ = -1;
  run_     = -1;
  evt_     = -1;
  runtype_ = -1;

  //this->subscribe();

  if( dataformat_client_ ) dataformat_client_->beginJob();
  if( digi_client_ )       digi_client_->beginJob();
  if( rechit_client_ )     rechit_client_->beginJob();
  if( pedestal_client_ )   pedestal_client_->beginJob(eventSetup);
  if( led_client_ )        led_client_->beginJob(eventSetup);
  if( hot_client_ )         hot_client_->beginJob();

  return;
}

void HcalMonitorClient::beginRun(void){

  if( verbose_ ) cout << "HcalMonitorClient: beginRun" << endl;
  begin_run_ = true;
  end_run_   = false;
  last_run_  = run_;

  this->setup();

  if( dataformat_client_ ) dataformat_client_->beginRun();
  if( digi_client_ )       digi_client_->beginRun();
  if( rechit_client_ )     rechit_client_->beginRun();
  if( pedestal_client_ )   pedestal_client_->beginRun();
  if( led_client_ )        led_client_->beginRun();
  if( hot_client_ )        hot_client_->beginRun();

  return;

}
void HcalMonitorClient::beginRun(const Run& r, const EventSetup& c) {

  cout << endl;
  cout << "Standard beginRun() for run " << r.id().run() << endl;
  cout << endl;

  if ( run_ != -1 && evt_ != -1 && runtype_ != -1 ) {
    if ( ! mergeRuns_ ) {
      forced_update_ = true;
      this->analyze();
      if ( ! begin_run_ ) {
        forced_status_ = false;
        this->beginRun();
      }
    }
  }
}

void HcalMonitorClient::endJob(void) {

  if( verbose_ ) cout << "HcalMonitorClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

  if( hot_client_ ) hot_client_->endJob();
  if( dataformat_client_ ) dataformat_client_->endJob();
  if( digi_client_ )  digi_client_->endJob();
  if( rechit_client_ )  rechit_client_->endJob();
  if( pedestal_client_ ) pedestal_client_->endJob();
  if( led_client_ ) led_client_->endJob();

  return;
}

///do a reset of all monitor elements...
void HcalMonitorClient::resetAllME() {
  if( dataformat_client_ ) dataformat_client_->resetAllME();
  if( digi_client_ )       digi_client_->resetAllME();
  if( rechit_client_ )     rechit_client_->resetAllME();
  if( pedestal_client_ )   pedestal_client_->resetAllME();
  if( led_client_ )        led_client_->resetAllME();
  if( hot_client_ )         hot_client_->resetAllME();
  return;
}


void HcalMonitorClient::report(bool doUpdate) {
  
  if( verbose_ ) cout << "HcalMonitorClient: creating report, ievt = " << ievt_ << endl;
  
  if(doUpdate && status_!="unknown" && mui_){
    this->createTests();  
    // mui_->update();
    //  mui_->doMonitoring();
    //  mui_->getBEInterface()->runQTests();
  }

  if( hot_client_ ) hot_client_->report();
  if( led_client_ ) led_client_->report();
  if( pedestal_client_ ) pedestal_client_->report();
  if( digi_client_ ) digi_client_->report();
  if( rechit_client_ ) rechit_client_->report();
  if( dataformat_client_ ) dataformat_client_->report();
  
  /*
  if(doUpdate && mui_){
    mui_->update();
    mui_->doMonitoring();
    mui_->getBEInterface()->runQTests();
  }
  */

  map<string, vector<QReport*> > errE, errW, errO;
  if( hot_client_ ) hot_client_->getErrors(errE,errW,errO);
  if( led_client_ ) led_client_->getErrors(errE,errW,errO);
  if( pedestal_client_ ) pedestal_client_->getErrors(errE,errW,errO);
  if( digi_client_ ) digi_client_->getErrors(errE,errW,errO);
  if( rechit_client_ ) rechit_client_->getErrors(errE,errW,errO);
  if( dataformat_client_ ) dataformat_client_->getErrors(errE,errW,errO);


  if( outputFile_.size() != 0) {    
    for( unsigned int i = 0; i < outputFile_.size(); i++ ) {
      if( outputFile_.substr(i, 5) == ".root" )  {
        outputFile_.replace(i, 5, "");
      }
    }
    char tmp[150];
    if(run_!=-1) sprintf(tmp,"%09d.root", run_);
    else sprintf(tmp,"%09d.root", 0);
    string fileName = outputFile_+tmp;
    mui_->getBEInterface()->save(fileName);
    
  }

  if( baseHtmlDir_.size() != 0 ) this->htmlOutput();

  status_="unknown";

  return;
}


void HcalMonitorClient::endRun(void) {
  if( verbose_ ) printf("HcalMonitorClient: endRun   updates: %d, events: %d\n",last_update_,mon_evt_);
  begin_run_ = false;
  end_run_   = true;

  printf("-->Creating report after run end condition\n");
  if(inputFile_.size()!=0) this->report(true);
  else this->report(false);
  
  if( hot_client_ )         hot_client_->endRun();
  if( dataformat_client_ ) dataformat_client_->endRun();
  if( digi_client_ )       digi_client_->endRun();
  if( rechit_client_ )     rechit_client_->endRun();
  if( pedestal_client_ )   pedestal_client_->endRun();
  if( led_client_ )        led_client_->endRun();
  
  //  this->cleanup();
  
  status_  = "unknown";
  runtype_ = "UNKNOWN";

  mon_evt_     = -1;  
  last_mon_evt_ = -1;
  last_update_ = -1;
  run_=-1;
  
  // this is an effective way to avoid ROOT memory leaks ...
  if( enableExit_ ) {
    cout << endl;
    cout << ">>> exit after End-Of-Run <<<" << endl;
    cout << endl;
    
    this->endJob();
    throw cms::Exception("End of Job")
      << "HcalMonitorClient: Done processing...\n";
  }

  return;
}

void HcalMonitorClient::endRun(const Run& r, const EventSetup& c) {

  cout << endl;
  cout << "Standard endRun() for run " << r.id().run() << endl;
  cout << endl;

  if ( run_ != -1 && evt_ != -1 && runtype_ != -1 ) {
    if ( ! mergeRuns_ ) {
      forced_update_ = true;
      this->analyze();
      if ( begin_run_ && ! end_run_ ) {
        forced_status_ = false;
        this->endRun();
      }
    }
  }
}

void HcalMonitorClient::beginLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) {

}

void HcalMonitorClient::endLuminosityBlock(const LuminosityBlock &l, const EventSetup &c) {

  this->analyze();
  
  if ( outputFile_.size() != 0 ) {
    string fileName = outputFile_;
    for ( unsigned int i = 0; i < fileName.size(); i++ ) {
      if( fileName.substr(i, 9) == "RUNNUMBER" )  {
        char tmp[10];
        if ( run_ != -1 ) {
          sprintf(tmp,"%09d", run_);
        } else {
          sprintf(tmp,"%09d", 0);
        }
        fileName.replace(i, 9, tmp);
        sprintf(tmp,".%04d", l.id().luminosityBlock());
        fileName.insert(i+9, tmp, 5);
      }
    }
    mui_->getBEInterface()->save(fileName);
  }

}

void HcalMonitorClient::setup(void) {
  return;
}

void HcalMonitorClient::cleanup(void) {  
  //this->unsubscribe();
  return;
}

void HcalMonitorClient::subscribe(void){

  if( verbose_ ) cout << "HcalMonitorClient: subscribe" << endl;

    if(mui_){
    Char_t histo[150];
    //mui_->subscribe("*/HcalMonitor/STATUS");
    sprintf(histo, "%sHcalMonitor/STATUS",process_.c_str());
    MonitorElement* me = mui_->getBEInterface()->get(histo);
    if(!me) return;
  }


  // subscribe to monitorable matching pattern
  if(mui_){
    //mui_->subscribe("*/FU0_is_done");
    //mui_->subscribe("*/FU0_is_dead");
    //mui_->subscribe("*/HcalMonitor/STATUS");
    //mui_->subscribe("*/HcalMonitor/RUN NUMBER");
    //mui_->subscribe("*/HcalMonitor/EVT NUMBER");
    //mui_->subscribe("*/HcalMonitor/EVT MASK");
    //mui_->subscribe("*/HcalMonitor/RUN TYPE");
  }
  /*
  if( hot_client_ ) hot_client_->subscribe();
  if( dataformat_client_ ) dataformat_client_->subscribe();
  if( digi_client_ ) digi_client_->subscribe();
  if( rechit_client_ ) rechit_client_->subscribe();
  if( pedestal_client_ ) pedestal_client_->subscribe();  
  if( led_client_ ) led_client_->subscribe();
*/
  subscribed_ = true;

  return;
}

void HcalMonitorClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  if(mui_){
    //mui_->subscribeNew("*/FU0_is_done");
    //mui_->subscribeNew("*/FU0_is_dead");
    //mui_->subscribeNew("*/HcalMonitor/STATUS");
    //mui_->subscribeNew("*/HcalMonitor/RUN NUMBER");
    //mui_->subscribeNew("*/HcalMonitor/EVT NUMBER");
    //mui_->subscribeNew("*/HcalMonitor/EVT MASK");
    //mui_->subscribeNew("*/HcalMonitor/RUN TYPE");
  }
  /*
  if( dataformat_client_ ) dataformat_client_->subscribeNew();
  if( digi_client_ ) digi_client_->subscribeNew();
  if( rechit_client_ ) rechit_client_->subscribeNew();
  if( pedestal_client_ ) pedestal_client_->subscribeNew();  
  if( led_client_ ) led_client_->subscribeNew();
  if( hot_client_ ) hot_client_->subscribeNew();
  */
  return;
}

void HcalMonitorClient::unsubscribe(void) {

  if( verbose_ ) cout << "HcalMonitorClient: unsubscribe" << endl;

  // unsubscribe to all monitorable matching pattern
  if(mui_){
    //  mui_->unsubscribe("*/FU0_is_done");
    //  mui_->unsubscribe("*/FU0_is_dead");
    //  mui_->unsubscribe("*/HcalMonitor/STATUS");
    //  mui_->unsubscribe("*/HcalMonitor/RUN NUMBER");
    //  mui_->unsubscribe("*/HcalMonitor/EVT NUMBER");
    //  mui_->unsubscribe("*/HcalMonitor/EVT MASK");
    //  mui_->unsubscribe("*/HcalMonitor/RUN TYPE");
  }
  /*
  if( hot_client_ ) hot_client_->unsubscribe();
  if( dataformat_client_ ) dataformat_client_->unsubscribe();
  if( digi_client_ ) digi_client_->unsubscribe();
  if( rechit_client_ ) rechit_client_->unsubscribe();
  if( pedestal_client_ ) pedestal_client_->unsubscribe();  
  if( led_client_ ) led_client_->unsubscribe();
  */
  subscribed_ = false;

  return;
}

void HcalMonitorClient::analyze(const Event& e, const edm::EventSetup& eventSetup){

  run_ = e.id().run();
  evt_ = e.id().event();
  
  this->analyze();
}


void HcalMonitorClient::analyze(){
  
  ievt_++;
  
  printf("\nHcal Monitor Client heartbeat....\n");
  if(!mui_){
    printf("HcalMonitorClient:  MonitorUserInterface NULL!!\n");
    return;
  }

  Char_t histo[150];
  MonitorElement* me =0;
  string s;  

  bool force_update = false;
  int updates = 1;
  //mui_->getBEInterface()->runQTests();
  /*
  // # of full monitoring cycles processed
  int updates = mui_->getNumUpdates();
  cout << " updates = " << updates << endl;
  printf("A2\n");
  printf("A3\n");
  mui_->doMonitoring();
  mui_->getBEInterface()->runQTests();

  printf("B\n");

  if(nTimeouts_>=timeoutThresh_ || inputFile_.size()!=0){
    if(verbose_) printf("\n\n\n\nHcalMonitorClient: Forcing update after timeout!\n\n\n\n");
    force_update = true;
    status_ = "end-of-run";
  }
  printf("C\n");

  //if no collector updates, continue....unless we're forcing an update
  if( updates != last_update_ || force_update) {
  */
  
  int lastRun = run_;
  
  sprintf(histo, "%sHcalMonitor/RUN NUMBER",process_.c_str());
  me = mui_->getBEInterface()->get(histo);    
  if( me ) {
    s = me->valueString();
    run_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &run_);
    if( verbose_ ) cout << "Found '" << histo << "'" << endl;
  }
  
  sprintf(histo, "%sHcalMonitor/EVT NUMBER",process_.c_str());
  me = mui_->getBEInterface()->get(histo);
  if( me ) {
    s = me->valueString();
    mon_evt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &mon_evt_);
    if( verbose_ ) cout << "Found '" << histo << "'" << endl;
  }
  
  sprintf(histo, "%sHcalMonitor/EVT MASK",process_.c_str());
  me = mui_->getBEInterface()->get(histo);
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
  
  if ( ! mergeRuns_ && run_ != last_run_ ) forced_update_ = true;
  
  
  if(!force_update){
    status_="unknown";
    sprintf(histo, "%sHcalMonitor/STATUS",process_.c_str());
    me = mui_->getBEInterface()->get(histo);
    if( me ) {
      s = me->valueString();
      status_ = "unknown";
      if( s.substr(2,1) == "0" ) status_ = "begin-of-run";
      if( s.substr(2,1) == "1" ) status_ = "running";
      if( s.substr(2,1) == "2" ) status_ = "end-of-run";
      if( verbose_ ) cout << "Found '" << histo << "'" << endl;
    }
  }
  
  if(verbose_) printf("HcalClient: run: %d, evts: %d, type: %s, status: %s, iter: %d, updates: %d\n",
		      run_, mon_evt_, runtype_.c_str(),status_.c_str(),ievt_, updates);
  
  
  ///check status of monitor
  if(status_=="begin-of-run") this->beginRun();
  else if(status_=="running"){
    if( dataformat_client_ ) dataformat_client_->analyze(); 	
    if( digi_client_ )       digi_client_->analyze(); 
    if( rechit_client_ )     rechit_client_->analyze(); 
    if( pedestal_client_ )   pedestal_client_->analyze();      
    if( led_client_ )        led_client_->analyze(); 
    if( hot_client_ )         hot_client_->analyze(); 
  }    
  else if(status_ == "end-of-run") this->endRun();    
  
  if(status_!="unknown" && (ievt_%10)==0 ){
    if((ievt_%nUpdateEvents_)!=0 || status_ == "begin-of-run"){ 
      createTests();  
      //    mui_->getBEInterface()->runQTests();
      //    mui_->update();
    }
  }

  //report triggers
  if(status_!="unknown"){
    if(run_!=lastRun && lastRun!=0 && mon_evt_>1){
      printf("-->Creating report after run transition\n");
      this->report(false);    
      last_mon_evt_ = mon_evt_;
    }
  }
  
  int addEvts = mon_evt_ - last_mon_evt_;
  if(addEvts>=nUpdateEvents_){
    printf("-->Creating report after %d events!\n",mon_evt_);
    this->report(true);
    last_mon_evt_ = mon_evt_;
  }
  
  nTimeouts_=0;
  //
  //else nTimeouts_++;
  
  last_update_ = updates;
  
  ///histogram reset functions
  if(resetUpdate_!=-1 && updates>0){ 
    if((updates % resetUpdate_) == 0){
      printf("-->Resetting histograms after %d updates!\n",updates);
      this->resetAllME();
    }
  }
  if(resetEvents_!=-1 && mon_evt_>0){ 
    int nSeenEvts = mon_evt_ - last_reset_Evts_;
    if(nSeenEvts>=resetEvents_){
      printf("-->Resetting histograms after %d events!\n",mon_evt_);    
      this->resetAllME();
      last_reset_Evts_ = mon_evt_;
    }
  }
  if(resetTime_!=-1){ 
    gettimeofday(&updateTime_,NULL);
    double deltaT=startTime_.tv_sec*1000.0+startTime_.tv_usec/1000.0;
    deltaT=updateTime_.tv_sec*1000.0+updateTime_.tv_usec/1000.0-deltaT;
    deltaT /= 1000.0; //convert to seconds...
    double nMin = deltaT/60.0;
    if(nMin>resetTime_){
      printf("-->Resetting histograms after %.2f minutes!\n",nMin);  
      this->resetAllME();    
      gettimeofday(&startTime_,NULL);
    }
  }
  
  // run number transition
  if ( status_ == "running" ) {
    if ( run_ != -1 && evt_ != -1 && runtype_ != -1) {
      if ( ! mergeRuns_ ) {
        int new_run_ = run_;
        int old_run_ = last_run_;

        if ( new_run_ != old_run_ ) {
          if ( begin_run_ && ! end_run_ ) {
            cout << endl;
            cout << " Old run has finished, issuing endRun() ... " << endl;
            cout << endl;

            // end old_run_
            run_ = old_run_;

            forced_status_ = false;
            this->endRun();
          }

          if ( ! begin_run_ ) {
            cout << endl;
            cout << " New run has started, issuing beginRun() ... " << endl;
            cout << endl;

            // start new_run_
            run_ = new_run_;

            forced_status_ = false;
            this->beginRun();
          }
        }
      }
    }
  }

  // 'running' state without a previous 'begin-of-run' state
  if ( status_ == "running" ) {
    if ( run_ != -1 && evt_ != -1 && runtype_ != -1 ) {
      if ( ! forced_status_ ) {
        if ( ! begin_run_ ) {
          cout << endl;
          cout << "Forcing beginRun() ... NOW !" << endl;
          cout << endl;

          forced_status_ = true;
          this->beginRun();
        }
      }
    }
  }
  // missing 'end-of-run' state, use the 'FU_is_done' ME
  if ( status_ == "running" ) {
    if ( run_ != -1 && evt_ != -1 && runtype_ != -1 ) {      
      if ( begin_run_ && ! end_run_ ) {	
        me = mui_->getBEInterface()->get("Collector/FU0_is_done");
        if ( me ) {
          cout << endl;
          cout << " Source FU0 is done, issuing endRun() ... " << endl;
          cout << endl;
	  
          forced_status_ = false;
          this->endRun();
        }
      }
    }
  }

  // missing 'end-of-run' state, use the 'FU_is_dead' ME  
  if ( status_ == "running" ) {
    if ( run_ != -1 && evt_ != -1 && runtype_ != -1 ) {
      if ( begin_run_ && ! end_run_ ) {
        me = mui_->getBEInterface()->get("Collector/FU0_is_dead");
        if ( me ) {
          cout << endl;
          cout << " Source FU0 is dead, issuing endRun() ... " << endl;
          cout << endl;

          forced_status_ = false;
          this->endRun();
        }
      }
    }
  }

  //this->subscribeNew();
  return;
}

void HcalMonitorClient::createTests(void){

  if( dataformat_client_ ) dataformat_client_->createTests(); 
  if( digi_client_ )       digi_client_->createTests(); 
  if( rechit_client_ )     rechit_client_->createTests(); 
  if( pedestal_client_ )   pedestal_client_->createTests(); 
  if( led_client_ )        led_client_->createTests(); 
  if( hot_client_ )        hot_client_->createTests(); 

  return;
}

void HcalMonitorClient::htmlOutput(void){

  cout << "Preparing HcalMonitorClient html output ..." << endl;

  char tmp[10];
  if(run_!=-1) sprintf(tmp, "%09d", run_);
  else sprintf(tmp, "%09d", 0);
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
  if( hot_client_ ) {
    htmlName = "HcalHotCellClient.html";
    hot_client_->htmlOutput(run_, htmlDir, htmlName);
    htmlFile << "<table border=0 WIDTH=\"50%\"><tr>" << endl;
    htmlFile << "<td WIDTH=\"35%\"><a href=\"" << htmlName << "\">Hot Cell Monitor</a></td" << endl;
    if(hot_client_->hasErrors()) htmlFile << "<td bgcolor=red align=center>This monitor task has errors.</td>" << endl;
    else if(hot_client_->hasWarnings()) htmlFile << "<td bgcolor=yellow align=center>This monitor task has warnings.</td>" << endl;
    else if(hot_client_->hasOther()) htmlFile << "<td bgcolor=aqua align=center>This monitor task has messages.</td>" << endl;
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
  
  if(trigger_!=NULL){
    if(trigger_->GetEntries()>0){
      htmlFile << "<hr>" << endl;
      htmlFile << "<h3><strong>Trigger Frequency</h3>" << endl;
      histoHTML(run_,trigger_,"Trigger Type","Evts", 100, htmlFile,htmlDir);  
    }
  }
  htmlFile << "</ul>" << endl;


  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();
  cout << "HcalMonitorClient html output done..." << endl;
  
  return;
}

void HcalMonitorClient::offlineSetup(){
  //  cout << endl;
  //  cout << " *** Hcal Generic Monitor Client, for offline operation***" << endl;
  //  cout << endl;

  dataformat_client_ = 0; digi_client_ = 0;
  rechit_client_ = 0; pedestal_client_ = 0;
  led_client_ = 0; 
  trigger_=0; hot_client_ = 0;
  
  begin_run_ = false;
  end_run_   = false;
  forced_status_ = false;
  forced_update_ = false;
  offline_ = true;

  status_  = "unknown"; runtype_ = "UNKNOWN";
  run_     = -1; mon_evt_     = -1;
  nTimeouts_ = 0;

  last_mon_evt_   = -1; last_update_ = 0;

  // base Html output directory
  baseHtmlDir_ = ".";
  
  // clients' constructors
  hot_client_          = new HcalHotCellClient();
  dataformat_client_   = new HcalDataFormatClient();
  rechit_client_       = new HcalRecHitClient();
  digi_client_         = new HcalDigiClient();
  pedestal_client_     = new HcalPedestalClient();
  led_client_          = new HcalLEDClient();

  return;
}

void HcalMonitorClient::loadHistograms(TFile* infile, const char* fname){
  
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
  
  trigger_ = (TH1F*)infile->Get("DQMData/HcalMonitor/TB Trigger Type");
  if(trigger_!=NULL) labelBins(trigger_);


  if(hot_client_) hot_client_->loadHistograms(infile);
  if(dataformat_client_) dataformat_client_->loadHistograms(infile);
  if(rechit_client_) rechit_client_->loadHistograms(infile);
  if(digi_client_) digi_client_->loadHistograms(infile);
  if(pedestal_client_) pedestal_client_->loadHistograms(infile);
  if(led_client_) led_client_->loadHistograms(infile);

  return;

}


void HcalMonitorClient::labelBins(TH1F* hist){
  
  if(hist==NULL) return;

  hist->GetXaxis()->SetBinLabel(1,"--");
  hist->GetXaxis()->SetBinLabel(2,"Beam Trigger");
  hist->GetXaxis()->SetBinLabel(3,"Out-Spill Ped");
  hist->GetXaxis()->SetBinLabel(4,"In-Spill Ped");
  hist->GetXaxis()->SetBinLabel(5,"LED Trigger");
  hist->GetXaxis()->SetBinLabel(6,"Laser Trigger");
  
  return;
}

void HcalMonitorClient::dumpHistograms(int& runNum, vector<TH1F*> &hist1d,vector<TH2F*> &hist2d){
  
  hist1d.clear(); 
  hist2d.clear(); 

  runNum = run_;

  /*
  if(hot_client_) hot_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(dataformat_client) dataformat_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(rechit_client_) rechit_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(digi_client_) digi_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(pedestal_client_) pedestal_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  if(led_client_) led_client_->dumpHistograms(names,meanX,meanY,rmsX,rmsY);
  */
  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include <DQM/HcalMonitorClient/interface/HcalMonitorClient.h>

DEFINE_FWK_MODULE(HcalMonitorClient);
