#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DQMEDHarvester::DQMEDHarvester() {
  usesResource("DQMStore");
}

void DQMEDHarvester::endJob() {
  DQMStore * store = edm::Service<DQMStore>().operator->();
  store->bookTransactionEndJob([this](DQMStore::IBooker &b, DQMStore::IGetter &g){
      this->dqmEndJob(b, g);
    });
}

void DQMEDHarvester::endLuminosityBlock(edm::LuminosityBlock const& iLumi,
					edm::EventSetup const& iSetup) {
  DQMStore * store = edm::Service<DQMStore>().operator->();
  store->bookTransactionEndLumi([this, &iLumi, &iSetup](DQMStore::IGetter &g){
      this->dqmEndLuminosityBlock(g, iLumi, iSetup);
    });
}
