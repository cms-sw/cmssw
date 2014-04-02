#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DQMEDHarvester::DQMEDHarvester() {
  usesResource("DQMStore");
}

void DQMEDHarvester::endJob() {
  DQMStore * store = edm::Service<DQMStore>().operator->();
  store->bookTransaction([this](DQMStore::IBooker &b, DQMStore::IGetter &g){
      this->manipulateHistograms(b, g);
    });
  dqmEndJob();
}
