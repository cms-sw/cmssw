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
#include <array>
#include <cassert>

// user include files
#include "FWCore/Framework/interface/stream/ProducingModuleAdaptorBase.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/EventForTransformer.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

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
    template <typename T>
    ProducingModuleAdaptorBase<T>::ProducingModuleAdaptorBase() {}

    template <typename T>
    ProducingModuleAdaptorBase<T>::~ProducingModuleAdaptorBase() {
      for (auto m : m_streamModules) {
        delete m;
      }
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::deleteModulesEarly() {
      for (auto m : m_streamModules) {
        delete m;
      }
      m_streamModules.clear();
    }

    //
    // member functions
    //

    template <typename T>
    void ProducingModuleAdaptorBase<T>::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      m_streamModules.resize(iPrealloc.numberOfStreams(), static_cast<T*>(nullptr));
      setupStreamModules();
      preallocRuns(iPrealloc.numberOfRuns());
      preallocLumis(iPrealloc.numberOfLuminosityBlocks());
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::registerProductsAndCallbacks(ProducingModuleAdaptorBase const*,
                                                                     ProductRegistry* reg) {
      auto firstMod = m_streamModules[0];
      if (firstMod->registrationCallback() and m_streamModules.size() > 1) {
        //we have a callback so we will collect all callbacks and create a new callback which calls them all.

        std::vector<std::function<void(BranchDescription const&)>> callbacks;
        callbacks.reserve(m_streamModules.size());

        for (auto mod : m_streamModules) {
          callbacks.push_back(mod->registrationCallback());
        }
        //Since only the first module will actually do the registration
        // we will change its callback to call all the callbacks
        firstMod->callWhenNewProductsRegistered([callbacks](BranchDescription const& iBD) {
          for (const auto& c : callbacks) {
            c(iBD);
          }
        });
      }
      firstMod->registerProducts(firstMod, reg, moduleDescription_);
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::itemsToGet(BranchType iType,
                                                   std::vector<ProductResolverIndexAndSkipBit>& iIndices) const {
      assert(not m_streamModules.empty());
      m_streamModules[0]->itemsToGet(iType, iIndices);
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::itemsMayGet(BranchType iType,
                                                    std::vector<ProductResolverIndexAndSkipBit>& iIndices) const {
      assert(not m_streamModules.empty());
      m_streamModules[0]->itemsMayGet(iType, iIndices);
    }

    template <typename T>
    std::vector<edm::ProductResolverIndexAndSkipBit> const& ProducingModuleAdaptorBase<T>::itemsToGetFrom(
        BranchType iType) const {
      assert(not m_streamModules.empty());
      return m_streamModules[0]->itemsToGetFrom(iType);
    }

    template <typename T>
    std::vector<ESProxyIndex> const& ProducingModuleAdaptorBase<T>::esGetTokenIndicesVector(
        edm::Transition iTrans) const {
      assert(not m_streamModules.empty());
      return m_streamModules[0]->esGetTokenIndicesVector(iTrans);
    }

    template <typename T>
    std::vector<ESRecordIndex> const& ProducingModuleAdaptorBase<T>::esGetTokenRecordIndicesVector(
        edm::Transition iTrans) const {
      assert(not m_streamModules.empty());
      return m_streamModules[0]->esGetTokenRecordIndicesVector(iTrans);
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::modulesWhoseProductsAreConsumed(
        std::array<std::vector<ModuleDescription const*>*, NumBranchTypes>& modules,
        std::vector<ModuleProcessName>& modulesInPreviousProcesses,
        ProductRegistry const& preg,
        std::map<std::string, ModuleDescription const*> const& labelsToDesc,
        std::string const& processName) const {
      assert(not m_streamModules.empty());
      return m_streamModules[0]->modulesWhoseProductsAreConsumed(
          modules, modulesInPreviousProcesses, preg, labelsToDesc, processName);
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::convertCurrentProcessAlias(std::string const& processName) {
      for (auto mod : m_streamModules) {
        mod->convertCurrentProcessAlias(processName);
      }
    }

    template <typename T>
    std::vector<edm::ConsumesInfo> ProducingModuleAdaptorBase<T>::consumesInfo() const {
      assert(not m_streamModules.empty());
      return m_streamModules[0]->consumesInfo();
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::updateLookup(BranchType iType,
                                                     ProductResolverIndexHelper const& iHelper,
                                                     bool iPrefetchMayGet) {
      for (auto mod : m_streamModules) {
        mod->updateLookup(iType, iHelper, iPrefetchMayGet);
      }
    }
    template <typename T>
    void ProducingModuleAdaptorBase<T>::updateLookup(eventsetup::ESRecordsToProxyIndices const& iPI) {
      for (auto mod : m_streamModules) {
        mod->updateLookup(iPI);
      }
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::resolvePutIndicies(BranchType iBranchType,
                                                           ModuleToResolverIndicies const& iIndicies,
                                                           std::string const& moduleLabel) {
      for (auto mod : m_streamModules) {
        mod->resolvePutIndicies(iBranchType, iIndicies, moduleLabel);
      }
    }

    template <typename T>
    std::vector<edm::ProductResolverIndex> const& ProducingModuleAdaptorBase<T>::indiciesForPutProducts(
        BranchType iBranchType) const {
      return m_streamModules[0]->indiciesForPutProducts(iBranchType);
    }

    template <typename T>
    ProductResolverIndex ProducingModuleAdaptorBase<T>::transformPrefetch_(size_t iTransformIndex) const {
      return 0;
    }
    template <typename T>
    size_t ProducingModuleAdaptorBase<T>::transformIndex_(edm::BranchDescription const& iBranch) const {
      return 0;
    }
    template <typename T>
    void ProducingModuleAdaptorBase<T>::doTransformAsync(WaitingTaskHolder iTask,
                                                         size_t iTransformIndex,
                                                         EventPrincipal const& iEvent,
                                                         ActivityRegistry*,
                                                         ModuleCallingContext const* iMCC,
                                                         ServiceWeakToken const&) {}

    template <typename T>
    void ProducingModuleAdaptorBase<T>::doBeginStream(StreamID id) {
      m_streamModules[id]->beginStream(id);
    }
    template <typename T>
    void ProducingModuleAdaptorBase<T>::doEndStream(StreamID id) {
      m_streamModules[id]->endStream();
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::doStreamBeginRun(StreamID id,
                                                         RunTransitionInfo const& info,
                                                         ModuleCallingContext const* mcc) {
      RunPrincipal const& rp = info.principal();
      auto mod = m_streamModules[id];
      setupRun(mod, rp.index());

      Run r(rp, moduleDescription_, mcc, false);
      r.setConsumer(mod);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::BeginRun), mod->esGetTokenIndices(Transition::BeginRun), parentC};
      mod->beginRun(r, c);
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::doStreamEndRun(StreamID id,
                                                       RunTransitionInfo const& info,
                                                       ModuleCallingContext const* mcc) {
      auto mod = m_streamModules[id];
      Run r(info, moduleDescription_, mcc, true);
      r.setConsumer(mod);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::EndRun), mod->esGetTokenIndices(Transition::EndRun), parentC};
      mod->endRun(r, c);
      streamEndRunSummary(mod, r, c);
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::doStreamBeginLuminosityBlock(StreamID id,
                                                                     LumiTransitionInfo const& info,
                                                                     ModuleCallingContext const* mcc) {
      LuminosityBlockPrincipal const& lbp = info.principal();
      auto mod = m_streamModules[id];
      setupLuminosityBlock(mod, lbp.index());

      LuminosityBlock lb(lbp, moduleDescription_, mcc, false);
      lb.setConsumer(mod);
      ESParentContext parentC(mcc);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                         mod->esGetTokenIndices(Transition::BeginLuminosityBlock),
                         parentC};
      mod->beginLuminosityBlock(lb, c);
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::doStreamEndLuminosityBlock(StreamID id,
                                                                   LumiTransitionInfo const& info,
                                                                   ModuleCallingContext const* mcc) {
      auto mod = m_streamModules[id];
      LuminosityBlock lb(info, moduleDescription_, mcc, true);
      lb.setConsumer(mod);
      ESParentContext parentC(mcc);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         mod->esGetTokenIndices(Transition::EndLuminosityBlock),
                         parentC};
      mod->endLuminosityBlock(lb, c);
      streamEndLuminosityBlockSummary(mod, lb, c);
    }

    template <typename T>
    void ProducingModuleAdaptorBase<T>::doRegisterThinnedAssociations(ProductRegistry const& registry,
                                                                      ThinnedAssociationsHelper& helper) {
      assert(not m_streamModules.empty());
      auto mod = m_streamModules[0];
      mod->registerThinnedAssociations(registry, helper);
    }
  }  // namespace stream
}  // namespace edm
