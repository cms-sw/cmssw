#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DQMEDAnalyzer::DQMEDAnalyzer() = default;

void DQMEDAnalyzer::beginRun(edm::Run const &iRun,
                             edm::EventSetup const &iSetup) {
  dqmBeginRun(iRun, iSetup);
  DQMStore * store = edm::Service<DQMStore>().operator->();
  store->bookTransaction([this, &iRun, &iSetup](DQMStore::IBooker &b) {
                           b.cd();
                           this->bookHistograms(b, iRun, iSetup);
                         },
                         iRun.run(),
                         0, 0);

}



void DQMEDAnalyzer::endLuminosityBlockSummary(edm::LuminosityBlock const &iLumi ,
                                              edm::EventSetup const &iSetup,
                                              dqmDetails::NoCache*) const {
  DQMStore * store = edm::Service<DQMStore>().operator->();
  assert(store);
  LogDebug("DQMEDAnalyzer") << "Merging Lumi local MEs ("
                            << iLumi.run() << ", "
                            << iLumi.id().luminosityBlock() << ", "
                            << stream_id_ << ", "
                            << iLumi.moduleCallingContext()->moduleDescription()->id()
                            << ") into the DQMStore@" << store << std::endl;
  store->mergeAndResetMEsLuminositySummaryCache(iLumi.run(),
                                                iLumi.id().luminosityBlock(),
                                                stream_id_,
                                                iLumi.moduleCallingContext()->moduleDescription()->id());
}

void DQMEDAnalyzer::endRunSummary(edm::Run const &iRun ,
                                  edm::EventSetup const &iSetup,
                                  dqmDetails::NoCache*) const {
  DQMStore * store = edm::Service<DQMStore>().operator->();
  assert(store);
  LogDebug("DQMEDAnalyzer") << "Merging Run local MEs ("
                            << iRun.run() << ", "
                            << stream_id_ << ", "
                            << iRun.moduleCallingContext()->moduleDescription()->id()
                            << ") into the DQMStore@" << store << std::endl;
  store->mergeAndResetMEsRunSummaryCache(iRun.run(),
                                         stream_id_,
                                         iRun.moduleCallingContext()->moduleDescription()->id());
}


void DQMEDAnalyzer::accumulate(edm::Event const& ev, edm::EventSetup const& es) {
  this->analyze(ev, es);
}
