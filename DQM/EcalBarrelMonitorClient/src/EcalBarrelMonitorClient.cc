/*
 * \file EcalBarrelMonitorClient.cc
 * 
 * $Date: 2005/11/09 14:57:58 $
 * $Revision: 1.0 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EcalBarrelMonitorClient.h>

EcalBarrelMonitorClient::EcalBarrelMonitorClient(const edm::ParameterSet& ps){

  laser_client = new EBMonitorLaserClient(ps);

  pedestal_client = new EBMonitorPedestalClient(ps);

}

EcalBarrelMonitorClient::~EcalBarrelMonitorClient(){

  delete laser_client;

  delete pedestal_client;

}

void EcalBarrelMonitorClient::beginJob(const edm::EventSetup& c){

  kevt = 0;

  laser_client->beginJob(c);

  pedestal_client->beginJob(c);

}

void EcalBarrelMonitorClient::endJob(void) {

  cout << "final kevt = " << kevt << endl;

  laser_client->endJob();

  pedestal_client->endJob();

}

void EcalBarrelMonitorClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  kevt++;
  cout << "kevt = " << kevt << endl;

  laser_client->analyze(e, c);

  pedestal_client->analyze(e, c);

}

