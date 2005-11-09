/*
 * \file EBMonitorLaserClient.cc
 * 
 * $Date: 2005/11/09 14:57:58 $
 * $Revision: 1.0 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBMonitorLaserClient.h>

EBMonitorLaserClient::EBMonitorLaserClient(const edm::ParameterSet& ps){

}

EBMonitorLaserClient::~EBMonitorLaserClient(){

}

void EBMonitorLaserClient::beginJob(const edm::EventSetup& c){

  jevt = 0;

}

void EBMonitorLaserClient::endJob(void) {

  cout << "final jevt = " << jevt << endl;

}

void EBMonitorLaserClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  jevt++;
  cout << "jevt = " << jevt << endl;

}

