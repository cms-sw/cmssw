#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DQMEDHarvester::DQMEDHarvester() {
  usesResource("DQMStore");
}

void DQMEDHarvester::beginRun(edm::Run const &iRun,
                             edm::EventSetup const &iSetup) {
}

void DQMEDHarvester::dqmEndRun(edm::Run const &iRun,
			       edm::EventSetup const &iSetup) {
}

void DQMEDHarvester::endRun(edm::Run const &iRun,
                             edm::EventSetup const &iSetup) {
  dqmEndRun(iRun, iSetup);
  DQMStore * store = edm::Service<DQMStore>().operator->();
  store->bookTransaction([this, &iRun, &iSetup](DQMStore::IBooker &b) {
      this->bookHistograms(b, iRun, iSetup);
    },
    iRun.run(),
    0,  //streamID
    0); //moduleID
}
