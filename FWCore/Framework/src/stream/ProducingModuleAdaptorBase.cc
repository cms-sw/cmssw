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
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"

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
    ProducingModuleAdaptorBase<T>::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      m_streamModules.resize(iPrealloc.numberOfStreams(),
                             static_cast<T*>(nullptr));
      setupStreamModules();
    }

    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::registerProductsAndCallbacks(ProducingModuleAdaptorBase const*, ProductRegistry* reg) {
      auto firstMod = m_streamModules[0];
      if(firstMod->registrationCallback() and m_streamModules.size()>1) {
        //we have a callback so we will collect all callbacks and create a new callback which calls them all.
        
        std::vector<std::function<void(BranchDescription const&)>> callbacks;
        callbacks.reserve(m_streamModules.size());
        
        for(auto mod: m_streamModules) {
          callbacks.push_back(mod->registrationCallback());
        }
        //Since only the first module will actually do the registration
        // we will change its callback to call all the callbacks
        firstMod->callWhenNewProductsRegistered([callbacks](BranchDescription const& iBD) {
          for(auto c: callbacks) {
            c(iBD);
          }
        });
      }
      firstMod->registerProducts(firstMod,reg,moduleDescription_);
    }
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::itemsToGet(BranchType iType, std::vector<ProductHolderIndexAndSkipBit>& iIndices) const {
      assert(not m_streamModules.empty());
      m_streamModules[0]->itemsToGet(iType,iIndices);
    }
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::itemsMayGet(BranchType iType, std::vector<ProductHolderIndexAndSkipBit>& iIndices) const {
      assert(not m_streamModules.empty());
      m_streamModules[0]->itemsMayGet(iType,iIndices);
    }

    template<typename T>
    std::vector<edm::ProductHolderIndexAndSkipBit> const&
    ProducingModuleAdaptorBase<T>::itemsToGetFromEvent() const {
      assert(not m_streamModules.empty());
      return m_streamModules[0]->itemsToGetFromEvent();
    }

    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::modulesDependentUpon(const std::string& iProcessName,
                                                        std::vector<const char*>& oModuleLabels) const {
      assert(not m_streamModules.empty());
      return m_streamModules[0]->modulesDependentUpon(iProcessName, oModuleLabels);
    }

    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::updateLookup(BranchType iType,
                                        ProductHolderIndexHelper const& iHelper) {
      for(auto mod: m_streamModules) {
        mod->updateLookup(iType,iHelper);
      }
    }

    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doBeginJob() {
      
    }
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doBeginStream(StreamID id) {
      m_streamModules[id]->beginStream(id);
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
                                                    ModuleCallingContext const* mcc)
    {
      auto mod = m_streamModules[id];
      setupRun(mod, rp.index());
      
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(mod);
      mod->beginRun(r, c);
      
    }    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doStreamEndRun(StreamID id,
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
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doStreamBeginLuminosityBlock(StreamID id,
                                                                LuminosityBlockPrincipal& lbp,
                                                                EventSetup const& c,
                                                                ModuleCallingContext const* mcc) {
      auto mod = m_streamModules[id];
      setupLuminosityBlock(mod,lbp.index());
      
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(mod);
      mod->beginLuminosityBlock(lb, c);
    }
    
    template< typename T>
    void
    ProducingModuleAdaptorBase<T>::doStreamEndLuminosityBlock(StreamID id,
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
