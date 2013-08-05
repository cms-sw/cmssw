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
#include "FWCore/Framework/src/CPCSentry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"


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
  m_streamModules.resize(1);
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
EDAnalyzerAdaptorBase::registerProductsAndCallbacks(EDAnalyzerAdaptorBase const*, ProductRegistry* reg) {
  for(auto mod : m_streamModules) {
    mod->registerProductsAndCallbacks(mod, reg);
  }
}



bool
EDAnalyzerAdaptorBase::doEvent(EventPrincipal& ep, EventSetup const& c,
                               CurrentProcessingContext const* cpcp) {
  assert(ep.streamID()<m_streamModules.size());
  auto mod = m_streamModules[ep.streamID()];
  detail::CPCSentry sentry(mod->current_context_, cpcp);
  Event e(ep, moduleDescription_);
  e.setConsumer(mod);
  mod->analyze(e, c);
  return true;
}
void
EDAnalyzerAdaptorBase::doBeginJob() {
  
}

void
EDAnalyzerAdaptorBase::doBeginStream(StreamID id) {
  m_streamModules[id]->beginStream();
}
void
EDAnalyzerAdaptorBase::doEndStream(StreamID id) {
  m_streamModules[id]->endStream();
}

void
EDAnalyzerAdaptorBase::doStreamBeginRun(StreamID id,
                                        RunPrincipal& rp,
                                        EventSetup const& c,
                                        CurrentProcessingContext const* cpcp)
{
  auto mod = m_streamModules[id];
  detail::CPCSentry sentry(mod->current_context_, cpcp);
  setupRun(mod, rp.index());
  
  Run r(rp, moduleDescription_);
  r.setConsumer(mod);
  mod->beginRun(r, c);

}

void
EDAnalyzerAdaptorBase::doStreamEndRun(StreamID id,
                    RunPrincipal& rp,
                    EventSetup const& c,
                    CurrentProcessingContext const* cpcp)
{
  auto mod = m_streamModules[id];
  detail::CPCSentry sentry(mod->current_context_, cpcp);
  Run r(rp, moduleDescription_);
  r.setConsumer(mod);
  mod->endRun(r, c);
  streamEndRunSummary(mod,r,c);
}

void
EDAnalyzerAdaptorBase::doStreamBeginLuminosityBlock(StreamID id,
                                                    LuminosityBlockPrincipal& lbp,
                                                    EventSetup const& c,
                                                    CurrentProcessingContext const* cpcp) {
  auto mod = m_streamModules[id];
  detail::CPCSentry sentry(mod->current_context_, cpcp);
  setupLuminosityBlock(mod,lbp.index());
  
  LuminosityBlock lb(lbp, moduleDescription_);
  lb.setConsumer(mod);
  mod->beginLuminosityBlock(lb, c);
}
void
EDAnalyzerAdaptorBase::doStreamEndLuminosityBlock(StreamID id,
                                LuminosityBlockPrincipal& lbp,
                                EventSetup const& c,
                                CurrentProcessingContext const* cpcp)
{
  auto mod = m_streamModules[id];
  detail::CPCSentry sentry(mod->current_context_, cpcp);
  LuminosityBlock lb(lbp, moduleDescription_);
  lb.setConsumer(this);
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
