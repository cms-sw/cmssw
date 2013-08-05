// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::stream::EDProducerAdaptorBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 21:43:44 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/EDProducerAdaptorBase.h"
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
EDProducerAdaptorBase::EDProducerAdaptorBase()
{
  m_streamModules.resize(1);
}

// EDProducerAdaptorBase::EDProducerAdaptorBase(const EDProducerAdaptorBase& rhs)
// {
//    // do actual copying here;
// }

EDProducerAdaptorBase::~EDProducerAdaptorBase()
{
  for(auto m: m_streamModules) {
    delete m;
  }
}

//
// assignment operators
//
// const EDProducerAdaptorBase& EDProducerAdaptorBase::operator=(const EDProducerAdaptorBase& rhs)
// {
//   //An exception safe implementation is
//   EDProducerAdaptorBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
EDProducerAdaptorBase::registerProductsAndCallbacks(EDProducerAdaptorBase const*, ProductRegistry* reg) {
  for(auto mod : m_streamModules) {
    //NOTE: this will cause us to register the same products multiple times
    // since each stream module will indepdently do this
    //Maybe I could only have module 0 do the registration and only call
    // the callbacks for the others
    mod->registerProducts(mod, reg, moduleDescription_);
  }
}



bool
EDProducerAdaptorBase::doEvent(EventPrincipal& ep, EventSetup const& c,
                               CurrentProcessingContext const* cpcp) {
  assert(ep.streamID()<m_streamModules.size());
  auto mod = m_streamModules[ep.streamID()];
  detail::CPCSentry sentry(mod->current_context_, cpcp);
  Event e(ep, moduleDescription_);
  e.setConsumer(mod);
  mod->produce(e, c);
  commit_(e,&mod->previousParentage_, &mod->previousParentageId_);
  return true;
}
void
EDProducerAdaptorBase::doBeginJob() {
  
}

void
EDProducerAdaptorBase::doBeginStream(StreamID id) {
  m_streamModules[id]->beginStream();
}
void
EDProducerAdaptorBase::doEndStream(StreamID id) {
  m_streamModules[id]->endStream();
}

void
EDProducerAdaptorBase::doStreamBeginRun(StreamID id,
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
EDProducerAdaptorBase::doStreamEndRun(StreamID id,
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
EDProducerAdaptorBase::doStreamBeginLuminosityBlock(StreamID id,
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
EDProducerAdaptorBase::doStreamEndLuminosityBlock(StreamID id,
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
EDProducerAdaptorBase::doRespondToOpenInputFile(FileBlock const& fb){}
void
EDProducerAdaptorBase::doRespondToCloseInputFile(FileBlock const& fb){}
void
EDProducerAdaptorBase::doPreForkReleaseResources(){}
void
EDProducerAdaptorBase::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren){}
