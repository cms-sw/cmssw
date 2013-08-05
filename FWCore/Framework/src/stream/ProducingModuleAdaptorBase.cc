// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::stream::ProducingModuleAdaptorBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 21:43:44 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/ProducingModuleAdaptorBase.h"
#include "FWCore/Framework/src/CPCSentry.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
namespace edm {
  namespace stream {
    template< typename T>
    ProducingModuleAdaptorBase<T>::ProducingModuleAdaptorBase()
    {
      m_streamModules.resize(1);
    }
    
    template< typename T>
    ProducingModuleAdaptorBase<T>::~ProducingModuleAdaptorBase()
    {
      for(auto m: m_streamModules) {
        delete m;
      }
    }
    
    //
    // member functions
    //
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::registerProductsAndCallbacks(ProducingModuleAdaptorBase const*, ProductRegistry* reg) {
      for(auto mod : m_streamModules) {
        //NOTE: this will cause us to register the same products multiple times
        // since each stream module will indepdently do this
        //Maybe I could only have module 0 do the registration and only call
        // the callbacks for the others
        mod->registerProducts(mod, reg, moduleDescription_);
      }
    }
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doBeginJob() {
      
    }
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doBeginStream(StreamID id) {
      m_streamModules[id]->beginStream();
    }
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doEndStream(StreamID id) {
      m_streamModules[id]->endStream();
    }
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doStreamBeginRun(StreamID id,
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
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doStreamEndRun(StreamID id,
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
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doStreamBeginLuminosityBlock(StreamID id,
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
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doStreamEndLuminosityBlock(StreamID id,
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
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doRespondToOpenInputFile(FileBlock const& fb){}
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doRespondToCloseInputFile(FileBlock const& fb){}
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doPreForkReleaseResources(){}
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren){}
  }
}