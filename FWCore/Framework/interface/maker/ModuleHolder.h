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
  class ThinnedAssociationsHelper;
  class ModuleConsumesInfo;
  struct ModuleConsumesMinimalESInfo;
  namespace maker {
    class ModuleHolder {
    public:
      ModuleHolder() = default;
      virtual ~ModuleHolder() {}
      virtual std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const* actions) const = 0;

      virtual ModuleDescription const& moduleDescription() const = 0;
      virtual std::vector<ModuleConsumesInfo> moduleConsumesInfos() const = 0;
      virtual std::vector<ModuleConsumesMinimalESInfo> moduleConsumesMinimalESInfos() const = 0;

      enum class Type { kAnalyzer, kFilter, kProducer, kOutputModule };
      enum class Concurrency { kGlobal, kLimited, kOne, kStream };

      virtual Type moduleType() const = 0;
      virtual Concurrency moduleConcurrencyType() const = 0;

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

      void registerThinnedAssociations(ProductRegistry const& registry, ThinnedAssociationsHelper& helper);
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
      virtual void implRegisterThinnedAssociations(ProductRegistry const& registry,
                                                   ThinnedAssociationsHelper& helper) = 0;
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
      std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const* actions) const final {
        return std::make_unique<edm::WorkerT<T>>(module(), moduleDescription(), actions);
      }

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
      void implRegisterThinnedAssociations(ProductRegistry const& registry, ThinnedAssociationsHelper& helper) final;

      void implRespondToOpenInputFile(FileBlock const& fb) final;
      void implRespondToCloseInputFile(FileBlock const& fb) final;
      void implRespondToCloseOutputFile() final;

      std::shared_ptr<T> m_mod;
    };
  }  // namespace maker
}  // namespace edm

#endif
