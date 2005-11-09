/*
 * \file EBMonitorPedestalClient.cc
 * 
 * $Date: 2005/11/09 17:29:05 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBMonitorPedestalClient.h>

EBMonitorPedestalClient::EBMonitorPedestalClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

}

EBMonitorPedestalClient::~EBMonitorPedestalClient(){

}

void EBMonitorPedestalClient::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

}

void EBMonitorPedestalClient::endJob(void) {

  cout << "EBMonitorPedestalClient final ievt = " << ievt_ << endl;

}

void EBMonitorPedestalClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  cout << "EBMonitorPedestalClient ievt = " << ievt_ << endl;

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

}

