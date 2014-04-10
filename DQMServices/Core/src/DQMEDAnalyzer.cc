#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DQMEDAnalyzer::DQMEDAnalyzer() {}

void DQMEDAnalyzer::beginStream(edm::StreamID id)
{
  stream_id_ = id.value();
}

void DQMEDAnalyzer::beginRun(edm::Run const &iRun,
                             edm::EventSetup const &iSetup) {
  dqmBeginRun(iRun, iSetup);
  DQMStore * store = edm::Service<DQMStore>().operator->();
  store->bookTransaction([this, &iRun, &iSetup](DQMStore::IBooker &b) {
                           this->bookHistograms(b, iRun, iSetup);
                         },
                         iRun.run(),
                         streamId(),
                         iRun.moduleCallingContext()->moduleDescription()->id());
}


std::shared_ptr<dqmDetails::NoCache>
DQMEDAnalyzer::globalBeginRunSummary(edm::Run const&,
                                     edm::EventSetup const&,
                                     RunContext const*)
{
  return nullptr;
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

void DQMEDAnalyzer::globalEndRunSummary(edm::Run const&,
                                        edm::EventSetup const&,
                                        RunContext const*,
                                        dqmDetails::NoCache*)
{}

std::shared_ptr<dqmDetails::NoCache>
DQMEDAnalyzer::globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                 edm::EventSetup const&,
                                                 LuminosityBlockContext const*)
{
  return nullptr;
}

void DQMEDAnalyzer::globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                    edm::EventSetup const&,
                                                    LuminosityBlockContext const*,
                                                    dqmDetails::NoCache*)
{}



//############################## ONLY NEEDED IN THE TRANSITION PERIOD ################################
//here the thread_unsafe (simplified) carbon copy of the DQMEDAnalyzer

thread_unsafe::DQMEDAnalyzer::DQMEDAnalyzer() {}

void thread_unsafe::DQMEDAnalyzer::beginRun(edm::Run const &iRun,
                             edm::EventSetup const &iSetup) {
  dqmBeginRun(iRun, iSetup);
  DQMStore * store = edm::Service<DQMStore>().operator->();
  store->bookTransaction([this, &iRun, &iSetup](DQMStore::IBooker &b) {
                           this->bookHistograms(b, iRun, iSetup);
                         },
                         0,
                         0,
                         0);
}

