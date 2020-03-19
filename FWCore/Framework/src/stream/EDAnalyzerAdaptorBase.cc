// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::stream::EDAnalyzerAdaptorBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 21:43:44 GMT
//

// system include files
#include <cassert>

// user include files
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"

using namespace edm::stream;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EDAnalyzerAdaptorBase::EDAnalyzerAdaptorBase() {}

// EDAnalyzerAdaptorBase::EDAnalyzerAdaptorBase(const EDAnalyzerAdaptorBase& rhs)
// {
//    // do actual copying here;
// }

EDAnalyzerAdaptorBase::~EDAnalyzerAdaptorBase() {
  for (auto m : m_streamModules) {
    delete m;
  }
}

//
// assignment operators
//
// const EDAnalyzerAdaptorBase& EDAnalyzerAdaptorBase::operator=(const EDAnalyzerAdaptorBase& rhs)
// {
//   //An exception safe implementation is
//   EDAnalyzerAdaptorBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void EDAnalyzerAdaptorBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
  m_streamModules.resize(iPrealloc.numberOfStreams(), static_cast<stream::EDAnalyzerBase*>(nullptr));
  setupStreamModules();
  preallocLumis(iPrealloc.numberOfLuminosityBlocks());
}

void EDAnalyzerAdaptorBase::registerProductsAndCallbacks(EDAnalyzerAdaptorBase const*, ProductRegistry* reg) {
  for (auto mod : m_streamModules) {
    mod->registerProductsAndCallbacks(mod, reg);
  }
}

void EDAnalyzerAdaptorBase::itemsToGet(BranchType iType, std::vector<ProductResolverIndexAndSkipBit>& iIndices) const {
  assert(not m_streamModules.empty());
  m_streamModules[0]->itemsToGet(iType, iIndices);
}
void EDAnalyzerAdaptorBase::itemsMayGet(BranchType iType, std::vector<ProductResolverIndexAndSkipBit>& iIndices) const {
  assert(not m_streamModules.empty());
  m_streamModules[0]->itemsMayGet(iType, iIndices);
}

std::vector<edm::ProductResolverIndexAndSkipBit> const& EDAnalyzerAdaptorBase::itemsToGetFrom(BranchType iType) const {
  assert(not m_streamModules.empty());
  return m_streamModules[0]->itemsToGetFrom(iType);
}

void EDAnalyzerAdaptorBase::updateLookup(BranchType iType,
                                         ProductResolverIndexHelper const& iHelper,
                                         bool iPrefetchMayGet) {
  for (auto mod : m_streamModules) {
    mod->updateLookup(iType, iHelper, iPrefetchMayGet);
  }
}

void EDAnalyzerAdaptorBase::updateLookup(eventsetup::ESRecordsToProxyIndices const& iPI) {
  for (auto mod : m_streamModules) {
    mod->updateLookup(iPI);
  }
}

const edm::EDConsumerBase* EDAnalyzerAdaptorBase::consumer() const { return m_streamModules[0]; }

void EDAnalyzerAdaptorBase::modulesWhoseProductsAreConsumed(
    std::vector<ModuleDescription const*>& modules,
    ProductRegistry const& preg,
    std::map<std::string, ModuleDescription const*> const& labelsToDesc,
    std::string const& processName) const {
  assert(not m_streamModules.empty());
  return m_streamModules[0]->modulesWhoseProductsAreConsumed(modules, preg, labelsToDesc, processName);
}

void EDAnalyzerAdaptorBase::convertCurrentProcessAlias(std::string const& processName) {
  for (auto mod : m_streamModules) {
    mod->convertCurrentProcessAlias(processName);
  }
}

std::vector<edm::ConsumesInfo> EDAnalyzerAdaptorBase::consumesInfo() const {
  assert(not m_streamModules.empty());
  return m_streamModules[0]->consumesInfo();
}

bool EDAnalyzerAdaptorBase::doEvent(EventPrincipal const& ep,
                                    EventSetupImpl const& ci,
                                    ActivityRegistry* act,
                                    ModuleCallingContext const* mcc) {
  assert(ep.streamID() < m_streamModules.size());
  auto mod = m_streamModules[ep.streamID()];
  Event e(ep, moduleDescription_, mcc);
  e.setConsumer(mod);
  const EventSetup c{
      ci, static_cast<unsigned int>(Transition::Event), mod->esGetTokenIndices(Transition::Event), false};
  EventSignalsSentry sentry(act, mcc);
  mod->analyze(e, c);
  return true;
}
void EDAnalyzerAdaptorBase::doBeginJob() {}

void EDAnalyzerAdaptorBase::doBeginStream(StreamID id) { m_streamModules[id]->beginStream(id); }
void EDAnalyzerAdaptorBase::doEndStream(StreamID id) { m_streamModules[id]->endStream(); }

void EDAnalyzerAdaptorBase::doStreamBeginRun(StreamID id,
                                             RunPrincipal const& rp,
                                             EventSetupImpl const& ci,
                                             ModuleCallingContext const* mcc) {
  auto mod = m_streamModules[id];
  setupRun(mod, rp.index());

  Run r(rp, moduleDescription_, mcc, false);
  const EventSetup c{
      ci, static_cast<unsigned int>(Transition::BeginRun), mod->esGetTokenIndices(Transition::BeginRun), false};
  r.setConsumer(mod);
  mod->beginRun(r, c);
}

void EDAnalyzerAdaptorBase::doStreamEndRun(StreamID id,
                                           RunPrincipal const& rp,
                                           EventSetupImpl const& ci,
                                           ModuleCallingContext const* mcc) {
  auto mod = m_streamModules[id];
  Run r(rp, moduleDescription_, mcc, true);
  r.setConsumer(mod);
  const EventSetup c{
      ci, static_cast<unsigned int>(Transition::EndRun), mod->esGetTokenIndices(Transition::EndRun), false};
  mod->endRun(r, c);
  streamEndRunSummary(mod, r, c);
}

void EDAnalyzerAdaptorBase::doStreamBeginLuminosityBlock(StreamID id,
                                                         LuminosityBlockPrincipal const& lbp,
                                                         EventSetupImpl const& ci,
                                                         ModuleCallingContext const* mcc) {
  auto mod = m_streamModules[id];
  setupLuminosityBlock(mod, lbp.index());

  LuminosityBlock lb(lbp, moduleDescription_, mcc, false);
  lb.setConsumer(mod);
  const EventSetup c{ci,
                     static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                     mod->esGetTokenIndices(Transition::BeginLuminosityBlock),
                     false};
  mod->beginLuminosityBlock(lb, c);
}
void EDAnalyzerAdaptorBase::doStreamEndLuminosityBlock(StreamID id,
                                                       LuminosityBlockPrincipal const& lbp,
                                                       EventSetupImpl const& ci,
                                                       ModuleCallingContext const* mcc) {
  auto mod = m_streamModules[id];
  LuminosityBlock lb(lbp, moduleDescription_, mcc, true);
  lb.setConsumer(mod);
  const EventSetup c{ci,
                     static_cast<unsigned int>(Transition::EndLuminosityBlock),
                     mod->esGetTokenIndices(Transition::EndLuminosityBlock),
                     false};
  mod->endLuminosityBlock(lb, c);
  streamEndLuminosityBlockSummary(mod, lb, c);
}

void EDAnalyzerAdaptorBase::doRespondToOpenInputFile(FileBlock const&) {}
void EDAnalyzerAdaptorBase::doRespondToCloseInputFile(FileBlock const&) {}

void EDAnalyzerAdaptorBase::setModuleDescriptionPtr(EDAnalyzerBase* m) {
  m->setModuleDescriptionPtr(&moduleDescription_);
}
