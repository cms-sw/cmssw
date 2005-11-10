/*
 * \file EcalBarrelMonitorClient.cc
 * 
 * $Date: 2005/11/10 08:26:07 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EcalBarrelMonitorClient.h>

EcalBarrelMonitorClient::EcalBarrelMonitorClient(const edm::ParameterSet& ps){

  cout << endl;
  cout << " *** Ecal Barrel Generic Monitor Client ***" << endl;
  cout << endl;

  // default client name
  string clientName = ps.getUntrackedParameter<string>("clientName","EcalBarrelMonitorClient");

  // default collector host name
  string hostName = ps.getUntrackedParameter<string>("hostName","localhost");

  // default host port
  int hostPort = ps.getUntrackedParameter<int>("hostPort",9090);;

  cout << " Client " << clientName
       << " begins requesting monitoring from host " << hostName
       << " on port " << hostPort << endl;

  // start user interface instance
  mui_ = new MonitorUIRoot(hostName, hostPort, clientName);

  mui_->setVerbose(1);

  // will attempt to reconnect upon connection problems (w/ a 5-sec delay)
  mui_->setReconnectDelay(5);

  laser_client_ = new EBLaserClient(ps, mui_);
  pedestal_client_ = new EBPedestalClient(ps, mui_);

}

EcalBarrelMonitorClient::~EcalBarrelMonitorClient(){

  cout << "Exit ..." << endl;

  delete laser_client_;
  delete pedestal_client_;

  mui_->unsubscribe("*");

  usleep(100);

  delete mui_;

}

void EcalBarrelMonitorClient::beginJob(const edm::EventSetup& c){

  ievt_ = 0;
  jevt_ = 0;

  laser_client_->beginJob(c);
  pedestal_client_->beginJob(c);

}

void EcalBarrelMonitorClient::beginRun(const edm::EventSetup& c){
  
  jevt_ = 0;

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/STATUS");
  mui_->subscribe("*/EcalBarrel/RUN");
  mui_->subscribe("*/EcalBarrel/EVT");
  mui_->subscribe("*/EcalBarrel/RUNTYPE");

  laser_client_->beginRun(c);
  pedestal_client_->beginRun(c);

}

void EcalBarrelMonitorClient::endJob(void) {

  cout << "EcalBarrelMonitorClient: endJob, ievt = " << ievt_ << endl;

  laser_client_->endJob();
  pedestal_client_->endJob();

}

void EcalBarrelMonitorClient::endRun(void) {

  cout << "EcalBarrelMonitorClient: endRun, jevt = " << jevt_ << endl;

  mui_->save("EcalBarrelMonitorClient.root");

  try {
    cout << "Opening DB connection." << endl;
    econn_ = new EcalCondDBInterface("pccmsecdb.cern.ch", "ecalh4db",
                                     "test06", "oratest06");
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

  laser_client_->endRun(econn_, runiov_, runtag_);
  pedestal_client_->endRun(econn_, runiov_, runtag_);

  cout << "Closing DB connection." << endl;

  delete econn_;

  delete runiov_;
  delete runtag_;

}

void EcalBarrelMonitorClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  cout << "EcalBarrelMonitorClient ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  MonitorElement* me;

  string s;
  string status = "unknown";
  string run    = "unknown";
  string evt    = "unknown";
  string type   = "unknown";

  bool stay_in_loop = mui_->update();

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/STATUS");
  mui_->subscribeNew("*/EcalBarrel/RUN");
  mui_->subscribeNew("*/EcalBarrel/EVT");
  mui_->subscribeNew("*/EcalBarrel/RUNTYPE");

  // # of full monitoring cycles processed
  int last_update = -1;
  int updates = mui_->getNumUpdates();

  if ( stay_in_loop && updates != last_update ) {

    me = mui_->get("Collector/FU0/EcalBarrel/STATUS");
    if ( me ) {
      s = me->valueString();
      if ( s.substr(2,1) == "0" ) status = "begin-of-run";
      if ( s.substr(2,1) == "1" ) status = "running";
      if ( s.substr(2,1) == "2" ) status = "end-of-run";
    }

    me = mui_->get("Collector/FU0/EcalBarrel/RUN");
    if ( me ) {
      s = me->valueString();
      run = s.substr(2,s.length()-2);
    }

    me = mui_->get("Collector/FU0/EcalBarrel/EVT");
    if ( me ) {
      s = me->valueString();
      evt = s.substr(2,s.length()-2);
    }

    me = mui_->get("Collector/FU0/EcalBarrel/RUNTYPE");
    if ( me ) {
      s = me->valueString();
      if ( s.substr(2,1) == "0" ) type = "cosmic";
      if ( s.substr(2,1) == "1" ) type = "laser";
      if ( s.substr(2,1) == "2" ) type = "pedestal";
      if ( s.substr(2,1) == "3" ) type = "testpulse";
    }

    last_update = updates;

    cout << " status = " << status <<
            " run = "    << run    <<
            " event = "  << evt    <<
            " type = "   << type   << endl;

// tests on 'type' to be changed with test on type-me !!

    if ( status == "begin-of-run" ) {
      this->beginRun(c);
    } else if ( status == "running" ) {
      if ( updates % 50 == 0 ) {
        if ( type == "laser" ) laser_client_->analyze(e, c);
        pedestal_client_->analyze(e, c);
      }
    } else if ( status == "end-of-run" ) {
      if ( type == "laser" ) laser_client_->analyze(e, c);
      pedestal_client_->analyze(e, c);
      this->endRun();
    }

  }

  if ( updates % 100 == 0 ) {
    mui_->save("EcalBarrelMonitorClient.root");
  }

  sleep(2);

}

