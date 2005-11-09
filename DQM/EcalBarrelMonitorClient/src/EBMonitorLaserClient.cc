/*
 * \file EBMonitorLaserClient.cc
 * 
 * $Date: 2005/11/09 17:29:05 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBMonitorLaserClient.h>

EBMonitorLaserClient::EBMonitorLaserClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

}

EBMonitorLaserClient::~EBMonitorLaserClient(){

}

void EBMonitorLaserClient::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

}

void EBMonitorLaserClient::endJob(void) {

  cout << "EBMonitorLaserClient final ievt = " << ievt_ << endl;

}

void EBMonitorLaserClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  cout << "EBMonitorLaserClient ievt = " << ievt_ << endl;

}

