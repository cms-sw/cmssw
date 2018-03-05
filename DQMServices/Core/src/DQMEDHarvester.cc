#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

#include <memory>

DQMEDHarvester::DQMEDHarvester() {
  usesResource("DQMStore");
  lumiToken_ = produces<DQMToken,edm::Transition::EndLuminosityBlock>("endLumi");
  runToken_ = produces<DQMToken,edm::Transition::EndRun>("endRun");
}

void DQMEDHarvester::endJob() {
  DQMStore * store = edm::Service<DQMStore>().operator->();
  store->meBookerGetter([this](DQMStore::IBooker &b, DQMStore::IGetter &g) {
    this->dqmEndJob(b, g);
  });
}

void DQMEDHarvester::endLuminosityBlock(edm::LuminosityBlock const& iLumi,
					edm::EventSetup const& iSetup) {
  DQMStore * store = edm::Service<DQMStore>().operator->();
  store->meBookerGetter([this, &iLumi, &iSetup](DQMStore::IBooker &b, DQMStore::IGetter &g){
    this->dqmEndLuminosityBlock(b, g, iLumi, iSetup);
  });
}

void DQMEDHarvester::endLuminosityBlockProduce(edm::LuminosityBlock& iLumi, edm::EventSetup const&) {
  iLumi.put(lumiToken_, std::make_unique<DQMToken>());
}

void DQMEDHarvester::endRunProduce(edm::Run& run, edm::EventSetup const& setup) {
  run.put(runToken_, std::make_unique<DQMToken>());
}
