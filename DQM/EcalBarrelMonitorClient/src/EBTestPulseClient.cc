/*
 * \file EBTestPulseClient.cc
 * 
 * $Date: 2005/11/10 09:55:15 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h>

EBTestPulseClient::EBTestPulseClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

}

EBTestPulseClient::~EBTestPulseClient(){

  this->unsubscribe();

}

void EBTestPulseClient::beginJob(const edm::EventSetup& c){

  cout << "EBTestPulseClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  this->beginRun(c);

}

void EBTestPulseClient::beginRun(const edm::EventSetup& c){

  cout << "EBTestPulseClient: beginRun" << endl;

  jevt_ = 0;

}

void EBTestPulseClient::endJob(void) {

  cout << "EBTestPulseClient: endJob, ievt = " << ievt_ << endl;

}

void EBTestPulseClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBTestPulseClient: endRun, jevt = " << jevt_ << endl;

}

void EBTestPulseClient::subscribe(void){

  // subscribe to all monitorable matching pattern

}

void EBTestPulseClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern

}

void EBTestPulseClient::unsubscribe(void){

  // unsubscribe to all monitorable matching pattern

}

void EBTestPulseClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  cout << "EBTestPulseClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

}

