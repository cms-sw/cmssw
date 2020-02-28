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

using namespace edm::stream;
namespace edm {
  namespace stream {

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

    bool EDProducerAdaptorBase::doEvent(EventPrincipal const& ep,
                                        EventSetupImpl const& ci,
                                        ActivityRegistry* act,
                                        ModuleCallingContext const* mcc) {
      assert(ep.streamID() < m_streamModules.size());
      auto mod = m_streamModules[ep.streamID()];
      Event e(ep, moduleDescription(), mcc);
      e.setConsumer(mod);
      e.setProducer(mod, &mod->previousParentage_, &mod->gotBranchIDsFromAcquire_);
      EventSignalsSentry sentry(act, mcc);
      const EventSetup c{
          ci, static_cast<unsigned int>(Transition::Event), mod->esGetTokenIndices(Transition::Event), false};
      mod->produce(e, c);
      commit(e, &mod->previousParentageId_);
      return true;
    }

    void EDProducerAdaptorBase::doAcquire(EventPrincipal const& ep,
                                          EventSetupImpl const& ci,
                                          ActivityRegistry* act,
                                          ModuleCallingContext const* mcc,
                                          WaitingTaskWithArenaHolder& holder) {
      assert(ep.streamID() < m_streamModules.size());
      auto mod = m_streamModules[ep.streamID()];
      Event e(ep, moduleDescription(), mcc);
      e.setConsumer(mod);
      e.setProducerForAcquire(mod, nullptr, mod->gotBranchIDsFromAcquire_);
      EventAcquireSignalsSentry sentry(act, mcc);
      const EventSetup c{
          ci, static_cast<unsigned int>(Transition::Event), mod->esGetTokenIndices(Transition::Event), false};
      mod->doAcquire_(e, c, holder);
    }

    template class edm::stream::ProducingModuleAdaptorBase<edm::stream::EDProducerBase>;
  }  // namespace stream
}  // namespace edm
