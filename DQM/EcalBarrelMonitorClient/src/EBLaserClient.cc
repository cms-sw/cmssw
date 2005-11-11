/*
 * \file EBLaserClient.cc
 * 
 * $Date: 2005/11/10 16:45:05 $
 * $Revision: 1.4 $
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

//  this->beginRun(c);

}

void EBLaserClient::beginRun(const edm::EventSetup& c){

  cout << "EBLaserClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

}

void EBLaserClient::endJob(void) {

  cout << "EBLaserClient: endJob, ievt = " << ievt_ << endl;

}

void EBLaserClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBLaserClient: endRun, jevt = " << jevt_ << endl;

  this->htmlOutput();

}

void EBLaserClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM*");

}

void EBLaserClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM*");

}

void EBLaserClient::unsubscribe(void){

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM*");

}

void EBLaserClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 )  
  cout << "EBLaserClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

}

void EBLaserClient::htmlOutput(void){

}

