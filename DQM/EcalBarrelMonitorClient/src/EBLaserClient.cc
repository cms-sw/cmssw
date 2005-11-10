/*
 * \file EBLaserClient.cc
 * 
 * $Date: 2005/11/10 09:55:15 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>

EBLaserClient::EBLaserClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

}

EBLaserClient::~EBLaserClient(){

  this->unsubscribe();

}

void EBLaserClient::beginJob(const edm::EventSetup& c){

  cout << "EBLaserClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  this->beginRun(c);

}

void EBLaserClient::beginRun(const edm::EventSetup& c){

  cout << "EBLaserClient: beginRun" << endl;

  jevt_ = 0;

}

void EBLaserClient::endJob(void) {

  cout << "EBLaserClient: endJob, ievt = " << ievt_ << endl;

}

void EBLaserClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBLaserClient: endRun, jevt = " << jevt_ << endl;

}

void EBLaserClient::subscribe(void){

  // subscribe to all monitorable matching pattern

}

void EBLaserClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern

}

void EBLaserClient::unsubscribe(void){

  // unsubscribe to all monitorable matching pattern

}

void EBLaserClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  cout << "EBLaserClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

}

