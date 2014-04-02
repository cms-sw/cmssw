#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DQMEDHarvester::DQMEDHarvester() {}

void DQMEDHarvester::beginRun(edm::Run const &iRun,
                             edm::EventSetup const &iSetup) {
  // dqmBeginRun(iRun, iSetup);
  // DQMStore * store = edm::Service<DQMStore>().operator->();
  // store->bookTransaction([this, &iRun, &iSetup](DQMStore::IBooker &b) {
  //                          this->bookHistograms(b, iRun, iSetup);
  //                        },
  //                        iRun.run(),
  //                        streamId(),
  //                        iRun.moduleCallingContext()->moduleDescription()->id());
}

void DQMEDHarvester::endRun(edm::Run const &iRun,
                             edm::EventSetup const &iSetup) {
}
