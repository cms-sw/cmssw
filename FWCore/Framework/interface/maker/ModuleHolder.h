#ifndef FWCore_Framework_ModuleHolder_h
#define FWCore_Framework_ModuleHolder_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ModuleHolder
//
/**\class edm::maker::ModuleHolder ModuleHolder.h "FWCore/Framework/interface/maker/ModuleHolder.h"

 Description: Base class used to own a module for the framework

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 23 Aug 2013 17:47:04 GMT
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/maker/WorkerT.h"
#include "FWCore/Framework/interface/SignallingProductRegistryFiller.h"
#include "FWCore/Framework/interface/OutputModuleCommunicator.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/TransitionPhaseTypes.h"
#include "FWCore/Framework/interface/maker/ModuleAttributes.h"

#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"

#include <unordered_map>
#include <string>
// forward declarations
namespace edm {
  class ModuleDescription;
  class SignallingProductRegistryFiller;
  class ExceptionToActionTable;
  class PreallocationConfiguration;
  class ProductResolverIndexHelper;
  class ProductResolverIndexAndSkipBit;
  class ProductRegistry;
  class ModuleConsumesInfo;
  struct ModuleConsumesMinimalESInfo;
  namespace maker {
    class ModuleHolder {
    public:
      ModuleHolder() = default;
      virtual ~ModuleHolder() {}
      virtual std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const* actions,
                                                 TransitionInfoKey key,
                                                 TransitionPhaseType phase) const = 0;

      virtual ModuleDescription const& moduleDescription() const = 0;
      virtual std::vector<ModuleConsumesInfo> moduleConsumesInfos() const = 0;
      virtual std::vector<ModuleConsumesMinimalESInfo> moduleConsumesMinimalESInfos() const = 0;

      using Type = edm::modules::Type;
      using Concurrency = edm::modules::Concurrency;

      virtual Type moduleType() const = 0;
      virtual Concurrency moduleConcurrencyType() const = 0;

      bool wantsTransition(TransitionInfoKey key, TransitionPhaseType phase) const noexcept {
        if (key == edm::RunTransitionInfo::key()) {
          if (phase == TransitionPhaseType::Global) {
            return wantsGlobalRuns() or wantsWrites();
          }
          return wantsStreamRuns();
        }
        if (key == edm::LumiTransitionInfo::key()) {
          if (phase == TransitionPhaseType::Global) {
            return wantsGlobalLuminosityBlocks() or wantsWrites();
          }
          return wantsStreamLuminosityBlocks();
        }
        if (key == edm::ProcessBlockTransitionInfo::key()) {
          return (phase == TransitionPhaseType::Global) and wantsProcessBlocks();
        }
        if (key == edm::InputProcessBlockTransitionInfo::key()) {
          return (phase == TransitionPhaseType::Global) and wantsInputProcessBlocks();
        }
        if (key == edm::EventTransitionInfo::key()) {
          return true;
        }
        assert(false);
        return false;
      }
      virtual bool wantsProcessBlocks() const noexcept = 0;
      virtual bool wantsInputProcessBlocks() const noexcept = 0;
      virtual bool wantsGlobalRuns() const noexcept = 0;
      virtual bool wantsGlobalLuminosityBlocks() const noexcept = 0;
      virtual bool wantsStreamRuns() const noexcept = 0;
      virtual bool wantsStreamLuminosityBlocks() const noexcept = 0;
      virtual bool wantsWrites() const noexcept = 0;

      virtual void finishModuleInitialization(ModuleDescription const& iDesc,
                                              PreallocationConfiguration const& iPrealloc,
                                              SignallingProductRegistryFiller* iReg) = 0;
      virtual void replaceModuleFor(Worker*) const = 0;

      virtual void beginJob() = 0;
      virtual void endJob() = 0;
      virtual void beginStream(StreamID) = 0;
      virtual void endStream(StreamID) = 0;

      void respondToOpenInputFile(FileBlock const& fb) { implRespondToOpenInputFile(fb); }
      void respondToCloseInputFile(FileBlock const& fb) { implRespondToCloseInputFile(fb); }
      void respondToCloseOutputFile() { implRespondToCloseOutputFile(); }

      virtual std::unique_ptr<OutputModuleCommunicator> createOutputModuleCommunicator() = 0;

      //Used to make EDGetToken work
      virtual void updateLookup(BranchType iBranchType, ProductResolverIndexHelper const&) = 0;
      virtual void updateLookup(eventsetup::ESRecordsToProductResolverIndices const&) = 0;
      virtual void releaseMemoryPostLookupSignal() = 0;
      virtual void selectInputProcessBlocks(ProductRegistry const&, ProcessBlockHelperBase const&) = 0;
      virtual void resolvePutIndicies(
          BranchType iBranchType,
          std::unordered_multimap<std::string, std::tuple<TypeID const*, const char*, edm::ProductResolverIndex>> const&
              iIndicies) = 0;
      virtual void convertCurrentProcessAlias(std::string const& processName) = 0;

    private:
      virtual void implRespondToOpenInputFile(FileBlock const& fb) = 0;
      virtual void implRespondToCloseInputFile(FileBlock const& fb) = 0;
      virtual void implRespondToCloseOutputFile() = 0;
    };

    template <typename T>
    class ModuleHolderT : public ModuleHolder {
    public:
      ModuleHolderT(std::shared_ptr<T> iModule) : m_mod(iModule) {}
      ~ModuleHolderT() override {}
      std::shared_ptr<T> module() const { return m_mod; }
      void replaceModuleFor(Worker* iWorker) const override {
        auto w = dynamic_cast<WorkerT<T>*>(iWorker);
        assert(nullptr != w);
        w->setModule(m_mod);
      }
      std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const* actions,
                                         TransitionInfoKey key,
                                         TransitionPhaseType phase) const final {
        return std::make_unique<edm::WorkerT<T>>(module(), moduleDescription(), actions);
      }

      bool wantsProcessBlocks() const noexcept final { return m_mod->wantsProcessBlocks(); }
      bool wantsInputProcessBlocks() const noexcept final { return m_mod->wantsInputProcessBlocks(); }
      bool wantsGlobalRuns() const noexcept final { return m_mod->wantsGlobalRuns(); }
      bool wantsGlobalLuminosityBlocks() const noexcept final { return m_mod->wantsGlobalLuminosityBlocks(); }
      bool wantsStreamRuns() const noexcept final { return m_mod->wantsStreamRuns(); }
      bool wantsStreamLuminosityBlocks() const noexcept final { return m_mod->wantsStreamLuminosityBlocks(); }
      bool wantsWrites() const noexcept final { return m_mod->wantsWrites(); }

      static void finishModuleInitialization(T& iModule,
                                             ModuleDescription const& iDesc,
                                             PreallocationConfiguration const& iPrealloc,
                                             SignallingProductRegistryFiller* iReg) {
        iModule.setModuleDescription(iDesc);
        iModule.doPreallocate(iPrealloc);
        if (iReg) {
          iModule.registerProductsAndCallbacks(&iModule, iReg);
        }
      };
      ModuleDescription const& moduleDescription() const final { return m_mod->moduleDescription(); }
      std::vector<ModuleConsumesInfo> moduleConsumesInfos() const final;
      std::vector<ModuleConsumesMinimalESInfo> moduleConsumesMinimalESInfos() const final;
      Type moduleType() const final;
      Concurrency moduleConcurrencyType() const final;

      void finishModuleInitialization(ModuleDescription const& iDesc,
                                      PreallocationConfiguration const& iPrealloc,
                                      SignallingProductRegistryFiller* iReg) override {
        finishModuleInitialization(*m_mod, iDesc, iPrealloc, iReg);
      }
      std::unique_ptr<OutputModuleCommunicator> createOutputModuleCommunicator() final;

      void beginJob() final;
      void endJob() final;
      void beginStream(StreamID) final;
      void endStream(StreamID) final;

      void updateLookup(BranchType iBranchType, ProductResolverIndexHelper const&) final;
      void updateLookup(eventsetup::ESRecordsToProductResolverIndices const&) final;
      void releaseMemoryPostLookupSignal() final;
      void selectInputProcessBlocks(ProductRegistry const&, ProcessBlockHelperBase const&) final;
      void resolvePutIndicies(
          BranchType iBranchType,
          std::unordered_multimap<std::string, std::tuple<TypeID const*, const char*, edm::ProductResolverIndex>> const&
              iIndicies) final;
      void convertCurrentProcessAlias(std::string const& processName) final {
        m_mod->convertCurrentProcessAlias(processName);
      }

    private:
      void implRespondToOpenInputFile(FileBlock const& fb) final;
      void implRespondToCloseInputFile(FileBlock const& fb) final;
      void implRespondToCloseOutputFile() final;

      std::shared_ptr<T> m_mod;
    };
  }  // namespace maker
}  // namespace edm

#endif
