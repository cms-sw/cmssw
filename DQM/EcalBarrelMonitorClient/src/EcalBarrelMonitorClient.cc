/*
 * \file EcalBarrelMonitorClient.cc
 * 
 * $Date: 2005/11/11 14:25:31 $
 * $Revision: 1.12 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EcalBarrelMonitorClient.h>

EcalBarrelMonitorClient::EcalBarrelMonitorClient(const edm::ParameterSet& ps){

  cout << endl;
  cout << " *** Ecal Barrel Generic Monitor Client ***" << endl;
  cout << endl;

  // default client name
  clientName_ = ps.getUntrackedParameter<string>("clientName","EcalBarrelMonitorClient");

  // default collector host name
  hostName_ = ps.getUntrackedParameter<string>("hostName","localhost");

  // default host port
  hostPort_ = ps.getUntrackedParameter<int>("hostPort",9090);;

  cout << " Client " << clientName_
       << " begins requesting monitoring from host " << hostName_
       << " on port " << hostPort_ << endl;

  // start user interface instance
  mui_ = new MonitorUIRoot(hostName_, hostPort_, clientName_);

  mui_->setVerbose(1);

  // will attempt to reconnect upon connection problems (w/ a 5-sec delay)
  mui_->setReconnectDelay(5);

  integrity_client_ = new EBIntegrityClient(ps, mui_);
  laser_client_ = new EBLaserClient(ps, mui_);
  pedestal_client_ = new EBPedestalClient(ps, mui_);
  pedpresample_client_ = new EBPedPreSampleClient(ps, mui_);
  testpulse_client_ = new EBTestPulseClient(ps, mui_);

  // Ecal Cond DB
  dbName_ = ps.getUntrackedParameter<string>("dbName", "");
  dbHostName_ = ps.getUntrackedParameter<string>("dbHostName", "");
  dbUserName_ = ps.getUntrackedParameter<string>("dbUserName", "");
  dbPassword_ = ps.getUntrackedParameter<string>("dbPassword", "");

  cout << " dbName = " << dbName_
       << " dbHostName = " << dbHostName_
       << " dbUserName = " << dbUserName_
       << endl;

}

EcalBarrelMonitorClient::~EcalBarrelMonitorClient(){

  cout << "Exit ..." << endl;

  this->unsubscribe();

  delete integrity_client_;
  delete laser_client_;
  delete pedestal_client_;
  delete pedpresample_client_;
  delete testpulse_client_;

  usleep(100);

  delete mui_;

}

void EcalBarrelMonitorClient::beginJob(const edm::EventSetup& c){

  cout << "EcalBarrelMonitorClient: beginJob" << endl;

  ievt_ = 0;

  this->beginRun(c);

  last_update_ = -1;

  integrity_client_->beginJob(c);
  laser_client_->beginJob(c);
  pedestal_client_->beginJob(c);
  pedpresample_client_->beginJob(c);
  testpulse_client_->beginJob(c);

}

void EcalBarrelMonitorClient::beginRun(const edm::EventSetup& c){
  
  cout << "EcalBarrelMonitorClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

  status_  = "unknown";
  run_     = 0;
  evt_     = 0;
  runtype_ = "unknown";

  integrity_client_->beginRun(c);
  laser_client_->beginRun(c);
  pedestal_client_->beginRun(c);
  pedpresample_client_->beginRun(c);
  testpulse_client_->beginRun(c);

}

void EcalBarrelMonitorClient::endJob(void) {

  cout << "EcalBarrelMonitorClient: endJob, ievt = " << ievt_ << endl;

  integrity_client_->endJob();
  laser_client_->endJob();
  pedestal_client_->endJob();
  pedpresample_client_->endJob();
  testpulse_client_->endJob();

}

void EcalBarrelMonitorClient::endRun(void) {

  cout << "EcalBarrelMonitorClient: endRun, jevt = " << jevt_ << endl;

  this->htmlOutput();

  mui_->save("EcalBarrelMonitorClient.root");

  try {
    cout << "Opening DB connection." << endl;
    econn_ = new EcalCondDBInterface(dbHostName_, dbName_,
                                     dbUserName_, dbPassword_);
  } catch (runtime_error &e) {
    cerr << e.what() << endl;
    return;
  }

  // The objects necessary to identify a dataset
  runiov_ = new RunIOV();
  runtag_ = new RunTag();

  Tm startTm;

  // Set the beginning time
  startTm.setToCurrentGMTime();
  startTm.setToMicrosTime(startTm.microsTime());

  cout << "Setting run " << run_ << " start_time " << startTm.str() << endl;

  runiov_->setRunNumber(run_);
  runiov_->setRunStart(startTm);

  runtag_->setRunType(runtype_);
  runtag_->setLocation(location_);
  runtag_->setMonitoringVersion("version 1");

  integrity_client_->endRun(econn_, runiov_, runtag_);
  laser_client_->endRun(econn_, runiov_, runtag_);
  pedestal_client_->endRun(econn_, runiov_, runtag_);
  pedpresample_client_->endRun(econn_, runiov_, runtag_);
  testpulse_client_->endRun(econn_, runiov_, runtag_);

  cout << "Closing DB connection." << endl;

  delete econn_;

  delete runiov_;
  delete runtag_;

}

void EcalBarrelMonitorClient::subscribe(void){

  // subscribe to monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/STATUS");
  mui_->subscribe("*/EcalBarrel/RUN");
  mui_->subscribe("*/EcalBarrel/EVT");
  mui_->subscribe("*/EcalBarrel/EVTTYPE");
  mui_->subscribe("*/EcalBarrel/RUNTYPE");

}

void EcalBarrelMonitorClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/STATUS");
  mui_->subscribeNew("*/EcalBarrel/RUN");
  mui_->subscribeNew("*/EcalBarrel/EVT");
  mui_->subscribeNew("*/EcalBarrel/EVTTYPE");
  mui_->subscribeNew("*/EcalBarrel/RUNTYPE");

}

void EcalBarrelMonitorClient::unsubscribe(void) {

  // subscribe to all monitorable matching pattern 
  mui_->unsubscribe("*/EcalBarrel/STATUS");
  mui_->unsubscribe("*/EcalBarrel/RUN");
  mui_->unsubscribe("*/EcalBarrel/EVT");
  mui_->unsubscribe("*/EcalBarrel/EVTTYPE");
  mui_->unsubscribe("*/EcalBarrel/RUNTYPE");

}

void EcalBarrelMonitorClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )
  cout << "EcalBarrelMonitorClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  string s;

  bool stay_in_loop = mui_->update();

  this->subscribeNew();

  integrity_client_->subscribeNew();
  laser_client_->subscribeNew();
  pedestal_client_->subscribeNew();
  pedpresample_client_->subscribeNew();
  testpulse_client_->subscribeNew();

  // # of full monitoring cycles processed
  int updates = mui_->getNumUpdates();

  if ( stay_in_loop && updates != last_update_ ) {

    MonitorElement* me;

    me = mui_->get("Collector/FU0/EcalBarrel/STATUS");
    if ( me ) {
      s = me->valueString();
      if ( s.substr(2,1) == "0" ) status_ = "begin-of-run";
      if ( s.substr(2,1) == "1" ) status_ = "running";
      if ( s.substr(2,1) == "2" ) status_ = "end-of-run";
    }

    me = mui_->get("Collector/FU0/EcalBarrel/RUN");
    if ( me ) {
      s = me->valueString();
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &run_);
    }

    me = mui_->get("Collector/FU0/EcalBarrel/EVT");
    if ( me ) {
      s = me->valueString();
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &evt_);
    }

    TH1F* h = 0;

    me = mui_->get("Collector/FU0/EcalBarrel/EVTTYPE");
    if ( me ) {
      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        h = dynamic_cast<TH1F*> (ob->operator->());
      }
    }

    me = mui_->get("Collector/FU0/EcalBarrel/RUNTYPE");
    if ( me ) {
      s = me->valueString();
      if ( s.substr(2,1) == "0" ) runtype_ = "cosmic";
      if ( s.substr(2,1) == "1" ) runtype_ = "laser";
      if ( s.substr(2,1) == "2" ) runtype_ = "pedestal";
      if ( s.substr(2,1) == "3" ) runtype_ = "testpulse";
    }

    last_update_ = updates;

    location_ = "H4";

    cout << " run = "      << run_      <<
            " event = "    << evt_      << endl;
    cout << " updates = "  << updates   <<
            " status = "   << status_   <<
            " runtype  = " << runtype_  <<
            " location = " << location_ << endl;

    if ( h ) {
      cout << " event type = " << flush;
      for ( int i = 1; i <=10; i++ ) {
        cout << h->GetBinContent(i) << " " << flush;
      }
      cout << endl;
    }

    if ( status_ == "begin-of-run" ) {
      this->beginRun(c);
    } else if ( status_ == "running" ) {
      if ( updates != 0 && updates % 50 == 0 ) {
                                             integrity_client_->analyze(e, c);
        if ( h && h->GetBinContent(2) != 0 ) laser_client_->analyze(e, c);
        if ( h && h->GetBinContent(3) != 0 ) pedestal_client_->analyze(e, c);
                                             pedpresample_client_->analyze(e, c);
        if ( h && h->GetBinContent(4) != 0 ) testpulse_client_->analyze(e, c);
      }
    } else if ( status_ == "end-of-run" ) {
                                           integrity_client_->analyze(e, c);
      if ( h && h->GetBinContent(2) != 0 ) laser_client_->analyze(e, c);
      if ( h && h->GetBinContent(3) != 0 ) pedestal_client_->analyze(e, c);
                                           pedpresample_client_->analyze(e, c);
      if ( h && h->GetBinContent(4) != 0 ) testpulse_client_->analyze(e, c);
      this->endRun();
    }

    if ( updates != 0 && updates % 100 == 0 ) {
      mui_->save("EcalBarrelMonitorClient.root");
    }

  }

//  sleep(1);

}

void EcalBarrelMonitorClient::htmlOutput(void){

  cout << "Preparing html output ..." << flush;

  char htmlOutputDir[99] = "./";

  char command[99];
  sprintf(command, "mkdir %s/%09d", htmlOutputDir, run_);

  try {
    system(command);
    cout << " done !" << endl;
  } catch (runtime_error &e) {
    cerr << e.what() << endl;
  }
 
}

