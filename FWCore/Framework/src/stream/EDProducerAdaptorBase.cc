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
    EDProducerAdaptorBase::EDProducerAdaptorBase()
    {
    }
    
    bool
    EDProducerAdaptorBase::doEvent(EventPrincipal& ep, EventSetup const& c,
                                   CurrentProcessingContext const* cpcp,
                                   ModuleCallingContext const* mcc) {
      assert(ep.streamID()<m_streamModules.size());
      auto mod = m_streamModules[ep.streamID()];
      detail::CPCSentry sentry(mod->current_context_, cpcp);
      Event e(ep, moduleDescription(), mcc);
      e.setConsumer(mod);
      mod->produce(e, c);
      commit(e,&mod->previousParentage_, &mod->previousParentageId_);
      return true;
    }
    
    template class edm::stream::ProducingModuleAdaptorBase<edm::stream::EDProducerBase>;
  }
}

