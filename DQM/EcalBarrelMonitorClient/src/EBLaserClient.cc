/*
 * \file EBLaserClient.cc
 * 
 * $Date: 2005/11/10 08:26:07 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>

EBLaserClient::EBLaserClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

}

EBLaserClient::~EBLaserClient(){

}

void EBLaserClient::beginJob(const edm::EventSetup& c){

  ievt_ = 0;
  jevt_ = 0;

}

void EBLaserClient::beginRun(const edm::EventSetup& c){

  jevt_ = 0;

}

void EBLaserClient::endJob(void) {

  cout << "EBLaserClient: endJob, ievt = " << ievt_ << endl;

}

void EBLaserClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBLaserClient: endRun, jevt = " << jevt_ << endl;

}

void EBLaserClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  cout << "EBLaserClient ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

}

