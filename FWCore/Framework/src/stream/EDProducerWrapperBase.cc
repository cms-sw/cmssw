// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::stream::EDProducerWrapperBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 21:43:44 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/EDProducerWrapperBase.h"
#include "FWCore/Framework/interface/stream/EDProducerBase.h"
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
EDProducerWrapperBase::EDProducerWrapperBase()
{
  m_streamModules.resize(1);
}

// EDProducerWrapperBase::EDProducerWrapperBase(const EDProducerWrapperBase& rhs)
// {
//    // do actual copying here;
// }

EDProducerWrapperBase::~EDProducerWrapperBase()
{
  for(auto m: m_streamModules) {
    delete m;
  }
}

//
// assignment operators
//
// const EDProducerWrapperBase& EDProducerWrapperBase::operator=(const EDProducerWrapperBase& rhs)
// {
//   //An exception safe implementation is
//   EDProducerWrapperBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
bool
EDProducerWrapperBase::doEvent(EventPrincipal& ep, EventSetup const& c,
                               CurrentProcessingContext const* cpcp) {

  auto mod = m_streamModules[ep.streamID()];
  detail::CPCSentry sentry(mod->current_context_, cpcp);
  Event e(ep, moduleDescription_);
  e.setConsumer(mod);
  mod->produce(e, c);
  commit_(e,&mod->previousParentage_, &mod->previousParentageId_);
  return true;
}
void
EDProducerWrapperBase::doBeginJob() {
  
}

void
EDProducerWrapperBase::doBeginStream(StreamID id) {
  m_streamModules[id]->beginStream();
}
void
EDProducerWrapperBase::doEndStream(StreamID id) {
  m_streamModules[id]->endStream();
}

void
EDProducerWrapperBase::doStreamBeginRun(StreamID id,
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
EDProducerWrapperBase::doStreamEndRun(StreamID id,
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
EDProducerWrapperBase::doStreamBeginLuminosityBlock(StreamID id,
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
EDProducerWrapperBase::doStreamEndLuminosityBlock(StreamID id,
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
