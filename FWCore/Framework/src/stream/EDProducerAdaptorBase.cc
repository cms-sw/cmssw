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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/src/EventAcquireSignalsSentry.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/src/stream/ProducingModuleAdaptorBase.cc"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

using namespace edm::stream;
namespace edm {
  namespace stream {

    template <>
    ProductResolverIndex ProducingModuleAdaptorBase<edm::stream::EDProducerBase>::transformPrefetch_(
        size_t iTransformIndex) const {
      return m_streamModules[0]->transformPrefetch_(iTransformIndex);
    }
    template <>
    size_t ProducingModuleAdaptorBase<edm::stream::EDProducerBase>::transformIndex_(
        edm::BranchDescription const& iBranch) const {
      return m_streamModules[0]->transformIndex_(iBranch);
    }
    template <>
    void ProducingModuleAdaptorBase<edm::stream::EDProducerBase>::doTransformAsync(WaitingTaskHolder iTask,
                                                                                   size_t iTransformIndex,
                                                                                   EventPrincipal const& iEvent,
                                                                                   ActivityRegistry*,
                                                                                   ModuleCallingContext const* iMCC,
                                                                                   ServiceWeakToken const& iToken) {
      EventForTransformer ev(iEvent, iMCC);
      m_streamModules[iEvent.streamID()]->transformAsync_(iTask, iTransformIndex, ev, iToken);
    }

    //
    // constants, enums and typedefs
    //

    //
    // static data member definitions
    //

    //
    // constructors and destructor
    //
    EDProducerAdaptorBase::EDProducerAdaptorBase() {}

    bool EDProducerAdaptorBase::doEvent(EventTransitionInfo const& info,
                                        ActivityRegistry* act,
                                        ModuleCallingContext const* mcc) {
      EventPrincipal const& ep = info.principal();
      assert(ep.streamID() < m_streamModules.size());
      auto mod = m_streamModules[ep.streamID()];
      Event e(ep, moduleDescription(), mcc);
      e.setConsumer(mod);
      e.setProducer(mod, &mod->previousParentage_, &mod->gotBranchIDsFromAcquire_);
      EventSignalsSentry sentry(act, mcc);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::Event), mod->esGetTokenIndices(Transition::Event), parentC};
      mod->produce(e, c);
      commit(e, &mod->previousParentageId_);
      return true;
    }

    void EDProducerAdaptorBase::doAcquire(EventTransitionInfo const& info,
                                          ActivityRegistry* act,
                                          ModuleCallingContext const* mcc,
                                          WaitingTaskWithArenaHolder& holder) {
      EventPrincipal const& ep = info.principal();
      assert(ep.streamID() < m_streamModules.size());
      auto mod = m_streamModules[ep.streamID()];
      Event e(ep, moduleDescription(), mcc);
      e.setConsumer(mod);
      e.setProducerForAcquire(mod, nullptr, mod->gotBranchIDsFromAcquire_);
      EventAcquireSignalsSentry sentry(act, mcc);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::Event), mod->esGetTokenIndices(Transition::Event), parentC};
      mod->doAcquire_(e, c, holder);
    }

    template class edm::stream::ProducingModuleAdaptorBase<edm::stream::EDProducerBase>;
  }  // namespace stream
}  // namespace edm
