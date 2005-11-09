/*
 * \file EBMonitorPedestalClient.cc
 * 
 * $Date: 2005/11/09 14:57:58 $
 * $Revision: 1.0 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBMonitorPedestalClient.h>

EBMonitorPedestalClient::EBMonitorPedestalClient(const edm::ParameterSet& ps){

}

EBMonitorPedestalClient::~EBMonitorPedestalClient(){

}

void EBMonitorPedestalClient::beginJob(const edm::EventSetup& c){

  ievt = 0;

}

void EBMonitorPedestalClient::endJob(void) {

  cout << "final ievt = " << ievt << endl;

}

void EBMonitorPedestalClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt++;
  cout << "ievt = " << ievt << endl;

}

