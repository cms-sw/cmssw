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

// user include files
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

#include "FWCore/Framework/src/PreallocationConfiguration.h"

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
EDAnalyzerAdaptorBase::EDAnalyzerAdaptorBase()
{
}

// EDAnalyzerAdaptorBase::EDAnalyzerAdaptorBase(const EDAnalyzerAdaptorBase& rhs)
// {
//    // do actual copying here;
// }

EDAnalyzerAdaptorBase::~EDAnalyzerAdaptorBase()
{
  for(auto m: m_streamModules) {
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
void
EDAnalyzerAdaptorBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
  m_streamModules.resize(iPrealloc.numberOfStreams(),
                         static_cast<stream::EDAnalyzerBase*>(nullptr));
  setupStreamModules();
}

void
EDAnalyzerAdaptorBase::registerProductsAndCallbacks(EDAnalyzerAdaptorBase const*, ProductRegistry* reg) {
  for(auto mod : m_streamModules) {
    mod->registerProductsAndCallbacks(mod, reg);
  }
}

void
EDAnalyzerAdaptorBase::itemsToGet(BranchType iType, std::vector<ProductHolderIndexAndSkipBit>& iIndices) const {
  assert(not m_streamModules.empty());
  m_streamModules[0]->itemsToGet(iType,iIndices);
}
void
EDAnalyzerAdaptorBase::itemsMayGet(BranchType iType, std::vector<ProductHolderIndexAndSkipBit>& iIndices) const {
  assert(not m_streamModules.empty());
  m_streamModules[0]->itemsMayGet(iType,iIndices);  
}

std::vector<edm::ProductHolderIndexAndSkipBit> const&
EDAnalyzerAdaptorBase::itemsToGetFromEvent() const {
  assert(not m_streamModules.empty());
  return m_streamModules[0]->itemsToGetFromEvent();  
}

void
EDAnalyzerAdaptorBase::updateLookup(BranchType iType,
                                    ProductHolderIndexHelper const& iHelper) {
  for(auto mod: m_streamModules) {
    mod->updateLookup(iType,iHelper);
  }
}

const edm::EDConsumerBase*
EDAnalyzerAdaptorBase::consumer() const {
  return m_streamModules[0];
}

void
EDAnalyzerAdaptorBase::modulesDependentUpon(const std::string& iProcessName,
                                            std::vector<const char*>& oModuleLabels) const {
  assert(not m_streamModules.empty());
  return m_streamModules[0]->modulesDependentUpon(iProcessName, oModuleLabels);
}

bool
EDAnalyzerAdaptorBase::doEvent(EventPrincipal& ep, EventSetup const& c,
                               ModuleCallingContext const* mcc) {
  assert(ep.streamID()<m_streamModules.size());
  auto mod = m_streamModules[ep.streamID()];
  Event e(ep, moduleDescription_, mcc);
  e.setConsumer(mod);
  mod->analyze(e, c);
  return true;
}
void
EDAnalyzerAdaptorBase::doBeginJob() {
  
}

void
EDAnalyzerAdaptorBase::doBeginStream(StreamID id) {
  m_streamModules[id]->beginStream(id);
}
void
EDAnalyzerAdaptorBase::doEndStream(StreamID id) {
  m_streamModules[id]->endStream();
}

void
EDAnalyzerAdaptorBase::doStreamBeginRun(StreamID id,
                                        RunPrincipal& rp,
                                        EventSetup const& c,
                                        ModuleCallingContext const* mcc)
{
  auto mod = m_streamModules[id];
  setupRun(mod, rp.index());
  
  Run r(rp, moduleDescription_, mcc);
  r.setConsumer(mod);
  mod->beginRun(r, c);

}

void
EDAnalyzerAdaptorBase::doStreamEndRun(StreamID id,
                    RunPrincipal& rp,
                    EventSetup const& c,
                    ModuleCallingContext const* mcc)
{
  auto mod = m_streamModules[id];
  Run r(rp, moduleDescription_, mcc);
  r.setConsumer(mod);
  mod->endRun(r, c);
  streamEndRunSummary(mod,r,c);
}

void
EDAnalyzerAdaptorBase::doStreamBeginLuminosityBlock(StreamID id,
                                                    LuminosityBlockPrincipal& lbp,
                                                    EventSetup const& c,
                                                    ModuleCallingContext const* mcc) {
  auto mod = m_streamModules[id];
  setupLuminosityBlock(mod,lbp.index());
  
  LuminosityBlock lb(lbp, moduleDescription_, mcc);
  lb.setConsumer(mod);
  mod->beginLuminosityBlock(lb, c);
}
void
EDAnalyzerAdaptorBase::doStreamEndLuminosityBlock(StreamID id,
                                LuminosityBlockPrincipal& lbp,
                                EventSetup const& c,
                                ModuleCallingContext const* mcc)
{
  auto mod = m_streamModules[id];
  LuminosityBlock lb(lbp, moduleDescription_, mcc);
  lb.setConsumer(mod);
  mod->endLuminosityBlock(lb, c);
  streamEndLuminosityBlockSummary(mod,lb, c);

}

void
EDAnalyzerAdaptorBase::doRespondToOpenInputFile(FileBlock const& fb){}
void
EDAnalyzerAdaptorBase::doRespondToCloseInputFile(FileBlock const& fb){}
void
EDAnalyzerAdaptorBase::doPreForkReleaseResources(){}
void
EDAnalyzerAdaptorBase::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren){}
