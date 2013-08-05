// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::stream::EDFilterAdaptorBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 21:43:44 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/EDFilterAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDFilterBase.h"
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
    EDFilterAdaptorBase::EDFilterAdaptorBase()
    {
    }
    
    bool
    EDFilterAdaptorBase::doEvent(EventPrincipal& ep, EventSetup const& c,
                                   CurrentProcessingContext const* cpcp) {
      assert(ep.streamID()<m_streamModules.size());
      auto mod = m_streamModules[ep.streamID()];
      detail::CPCSentry sentry(mod->current_context_, cpcp);
      Event e(ep, moduleDescription());
      e.setConsumer(mod);
      bool result = mod->filter(e, c);
      commit(e,&mod->previousParentage_, &mod->previousParentageId_);
      return result;
    }
    
    template class edm::stream::ProducingModuleAdaptorBase<edm::stream::EDFilterBase>;
  }
}

