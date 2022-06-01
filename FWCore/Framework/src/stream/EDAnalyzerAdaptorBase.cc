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
#include <array>
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
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"

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

void EDAnalyzerAdaptorBase::deleteModulesEarly() {
  for (auto m : m_streamModules) {
    delete m;
  }
  m_streamModules.clear();
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

std::vector<edm::ESProxyIndex> const& EDAnalyzerAdaptorBase::esGetTokenIndicesVector(edm::Transition iTrans) const {
  assert(not m_streamModules.empty());
  return m_streamModules[0]->esGetTokenIndicesVector(iTrans);
}

std::vector<edm::ESRecordIndex> const& EDAnalyzerAdaptorBase::esGetTokenRecordIndicesVector(
    edm::Transition iTrans) const {
  assert(not m_streamModules.empty());
  return m_streamModules[0]->esGetTokenRecordIndicesVector(iTrans);
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
    std::array<std::vector<ModuleDescription const*>*, NumBranchTypes>& modules,
    std::vector<ModuleProcessName>& modulesInPreviousProcesses,
    ProductRegistry const& preg,
    std::map<std::string, ModuleDescription const*> const& labelsToDesc,
    std::string const& processName) const {
  assert(not m_streamModules.empty());
  return m_streamModules[0]->modulesWhoseProductsAreConsumed(
      modules, modulesInPreviousProcesses, preg, labelsToDesc, processName);
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

bool EDAnalyzerAdaptorBase::doEvent(EventTransitionInfo const& info,
                                    ActivityRegistry* act,
                                    ModuleCallingContext const* mcc) {
  EventPrincipal const& ep = info.principal();
  assert(ep.streamID() < m_streamModules.size());
  auto mod = m_streamModules[ep.streamID()];
  Event e(ep, moduleDescription_, mcc);
  e.setConsumer(mod);
  ESParentContext parentC(mcc);
  const EventSetup c{
      info, static_cast<unsigned int>(Transition::Event), mod->esGetTokenIndices(Transition::Event), parentC};
  EventSignalsSentry sentry(act, mcc);
  mod->analyze(e, c);
  return true;
}

void EDAnalyzerAdaptorBase::doBeginStream(StreamID id) { m_streamModules[id]->beginStream(id); }
void EDAnalyzerAdaptorBase::doEndStream(StreamID id) { m_streamModules[id]->endStream(); }

void EDAnalyzerAdaptorBase::doStreamBeginRun(StreamID id,
                                             RunTransitionInfo const& info,
                                             ModuleCallingContext const* mcc) {
  RunPrincipal const& rp = info.principal();
  auto mod = m_streamModules[id];
  setupRun(mod, rp.index());

  Run r(rp, moduleDescription_, mcc, false);
  ESParentContext parentC(mcc);
  const EventSetup c{
      info, static_cast<unsigned int>(Transition::BeginRun), mod->esGetTokenIndices(Transition::BeginRun), parentC};
  r.setConsumer(mod);
  mod->beginRun(r, c);
}

void EDAnalyzerAdaptorBase::doStreamEndRun(StreamID id,
                                           RunTransitionInfo const& info,
                                           ModuleCallingContext const* mcc) {
  auto mod = m_streamModules[id];
  Run r(info, moduleDescription_, mcc, true);
  r.setConsumer(mod);
  ESParentContext parentC(mcc);
  const EventSetup c{
      info, static_cast<unsigned int>(Transition::EndRun), mod->esGetTokenIndices(Transition::EndRun), parentC};
  mod->endRun(r, c);
  streamEndRunSummary(mod, r, c);
}

void EDAnalyzerAdaptorBase::doStreamBeginLuminosityBlock(StreamID id,
                                                         LumiTransitionInfo const& info,
                                                         ModuleCallingContext const* mcc) {
  LuminosityBlockPrincipal const& lbp = info.principal();
  auto mod = m_streamModules[id];
  setupLuminosityBlock(mod, lbp.index());

  LuminosityBlock lb(lbp, moduleDescription_, mcc, false);
  lb.setConsumer(mod);
  ESParentContext parentC(mcc);
  const EventSetup c{info,
                     static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                     mod->esGetTokenIndices(Transition::BeginLuminosityBlock),
                     parentC};
  mod->beginLuminosityBlock(lb, c);
}
void EDAnalyzerAdaptorBase::doStreamEndLuminosityBlock(StreamID id,
                                                       LumiTransitionInfo const& info,
                                                       ModuleCallingContext const* mcc) {
  auto mod = m_streamModules[id];
  LuminosityBlock lb(info, moduleDescription_, mcc, true);
  lb.setConsumer(mod);
  ESParentContext parentC(mcc);
  const EventSetup c{info,
                     static_cast<unsigned int>(Transition::EndLuminosityBlock),
                     mod->esGetTokenIndices(Transition::EndLuminosityBlock),
                     parentC};
  mod->endLuminosityBlock(lb, c);
  streamEndLuminosityBlockSummary(mod, lb, c);
}

void EDAnalyzerAdaptorBase::setModuleDescriptionPtr(EDAnalyzerBase* m) {
  m->setModuleDescriptionPtr(&moduleDescription_);
}
