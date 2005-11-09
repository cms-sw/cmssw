/*
 * \file EcalBarrelMonitorClient.cc
 * 
 * $Date: 2005/11/09 17:29:05 $
 * $Revision: 1.1 $
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

  laser_client_ = new EBMonitorLaserClient(ps, mui_);

  pedestal_client_ = new EBMonitorPedestalClient(ps, mui_);

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

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/STATUS");
  mui_->subscribe("*/EcalBarrel/RUN");
  mui_->subscribe("*/EcalBarrel/EVT");
  mui_->subscribe("*/EcalBarrel/RUNTYPE");

  laser_client_->beginJob(c);

  pedestal_client_->beginJob(c);

}

void EcalBarrelMonitorClient::endJob(void) {

  cout << "EcalBarrelMonitorClient final ievt = " << ievt_ << endl;

  laser_client_->endJob();

  pedestal_client_->endJob();

}

void EcalBarrelMonitorClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  cout << "EcalBarrelMonitorClient ievt = " << ievt_ << endl;

  MonitorElement* me;

  string s;
  string status = "unknown";
  string run    = "unknown";
  string evt    = "unknown";
  string type   = "unknown";

  bool stay_in_loop = mui_->update();

  cout << "stay_in_loop = " << stay_in_loop << endl;

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/STATUS");
  mui_->subscribeNew("*/EcalBarrel/RUN");
  mui_->subscribeNew("*/EcalBarrel/EVT");
  mui_->subscribeNew("*/EcalBarrel/RUNTYPE");

  // # of full monitoring cycles processed
  int last_update = -1;
  int updates = mui_->getNumUpdates();

  if ( updates != last_update ) {

    me = mui_->get("Collector/FU0/EcalBarrel/STATUS");
    if ( me ) {
      s = me->valueString();
      if ( s.substr(2,1) == "0" ) status = "start-of-run";
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
            " run = "   << run    <<
            " event = " << evt    <<
            " type = "  << type   << endl;
  }

  laser_client_->analyze(e, c);

  pedestal_client_->analyze(e, c);

  sleep(2);
}

