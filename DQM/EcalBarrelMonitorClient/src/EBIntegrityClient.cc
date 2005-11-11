/*
 * \file EBIntegrityClient.cc
 * 
 * $Date: 2005/11/11 14:25:31 $
 * $Revision: 1.4 $
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

//  this->beginRun(c);

}

void EBIntegrityClient::beginRun(const edm::EventSetup& c){

  cout << "EBIntegrityClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

}

void EBIntegrityClient::endJob(void) {

  cout << "EBIntegrityClient: endJob, ievt = " << ievt_ << endl;

}

void EBIntegrityClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBIntegrityClient: endRun, jevt = " << jevt_ << endl;

  this->htmlOutput();

}

void EBIntegrityClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalIntegrity/Gain/EI gain SM*");
  mui_->subscribe("*/EcalIntegrity/ChId/EI ChId SM*");
  mui_->subscribe("*/EcalIntegrity/TTId/EI TTId SM*");
  mui_->subscribe("*/EcalIntegrity/TTBlockSize/EI TTBlockSize SM*");
  mui_->subscribe("*/EcalIntegrity/DCC size error");

}

void EBIntegrityClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalIntegrity/Gain/EI gain SM*");
  mui_->subscribeNew("*/EcalIntegrity/ChId/EI ChId SM*");
  mui_->subscribeNew("*/EcalIntegrity/TTId/EI TTId SM*");
  mui_->subscribeNew("*/EcalIntegrity/TTBlockSize/EI TTBlockSize SM*");
  mui_->subscribeNew("*/EcalIntegrity/DCC size error");

}

void EBIntegrityClient::unsubscribe(void){
  
  // unsubscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalIntegrity/Gain/EI gain SM*");
  mui_->subscribe("*/EcalIntegrity/ChId/EI ChId SM*");
  mui_->subscribe("*/EcalIntegrity/TTId/EI TTId SM*");
  mui_->subscribe("*/EcalIntegrity/TTBlockSize/EI TTBlockSize SM*");
  mui_->subscribe("*/EcalIntegrity/DCC size error");

}

void EBIntegrityClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )
  cout << "EBIntegrityClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  this->subscribeNew();

}

void EBIntegrityClient::htmlOutput(void){

}

