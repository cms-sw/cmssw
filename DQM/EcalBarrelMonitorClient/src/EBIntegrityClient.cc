/*
 * \file EBIntegrityClient.cc
 * 
 * $Date: 2005/11/10 09:55:15 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h>

EBIntegrityClient::EBIntegrityClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

}

EBIntegrityClient::~EBIntegrityClient(){

  this->unsubscribe();

}

void EBIntegrityClient::beginJob(const edm::EventSetup& c){

  cout << "EBIntegrityClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  this->beginRun(c);

}

void EBIntegrityClient::beginRun(const edm::EventSetup& c){

  cout << "EBIntegrityClient: beginRun" << endl;

  jevt_ = 0;

}

void EBIntegrityClient::endJob(void) {

  cout << "EBIntegrityClient: endJob, ievt = " << ievt_ << endl;

}

void EBIntegrityClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBIntegrityClient: endRun, jevt = " << jevt_ << endl;

}

void EBIntegrityClient::subscribe(void){

  // subscribe to all monitorable matching pattern

}

void EBIntegrityClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern

}

void EBIntegrityClient::unsubscribe(void){
  
  // unsubscribe to all monitorable matching pattern

}

void EBIntegrityClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  cout << "EBIntegrityClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

}

