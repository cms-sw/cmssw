#include "FWCore/Framework/interface/maker/ModuleHolder.h"
#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/one/EDFilterBase.h"
#include "FWCore/Framework/interface/one/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/one/OutputModuleBase.h"
#include "FWCore/Framework/interface/global/EDProducerBase.h"
#include "FWCore/Framework/interface/global/EDFilterBase.h"
#include "FWCore/Framework/interface/global/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/global/OutputModuleBase.h"

#include "FWCore/Framework/interface/stream/EDProducerAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDFilterAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptorBase.h"

#include "FWCore/Framework/interface/limited/EDProducerBase.h"
#include "FWCore/Framework/interface/limited/EDFilterBase.h"
#include "FWCore/Framework/interface/limited/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/limited/OutputModuleBase.h"

#include "FWCore/Framework/interface/OutputModuleCommunicatorT.h"

namespace edm::maker {
  namespace {
    template <typename T>
    concept HasRegisterThinnedAssociationsFunction =
        requires(T mod, ProductRegistry reg, ThinnedAssociationsHelper helper) {
          { mod.doRegisterThinnedAssociations(reg, helper) } -> std::same_as<void>;
        };
  }  // namespace

  template <typename T>
  inline void ModuleHolderT<T>::implRegisterThinnedAssociations(ProductRegistry const& registry,
                                                                ThinnedAssociationsHelper& helper) {
    if constexpr (HasRegisterThinnedAssociationsFunction<T>) {
      m_mod->doRegisterThinnedAssociations(registry, helper);
    }
  }

  template <typename T>
  std::unique_ptr<OutputModuleCommunicator> ModuleHolderT<T>::createOutputModuleCommunicator() {
    return OutputModuleCommunicatorT<T>::createIfNeeded(m_mod.get());
  }

  namespace {
    template <typename T>
    concept HasStreamFunctions = requires(T mod, StreamID id) {
      { mod.doBeginStream(id) } -> std::same_as<void>;
      { mod.doEndStream(id) } -> std::same_as<void>;
    };
  }  // namespace

  template <typename T>
  void ModuleHolderT<T>::beginJob() {
    m_mod->doBeginJob();
  }
  template <typename T>
  void ModuleHolderT<T>::endJob() {
    m_mod->doEndJob();
  }
  template <typename T>
  void ModuleHolderT<T>::beginStream(StreamID iID) {
    if constexpr (HasStreamFunctions<T>) {
      m_mod->doBeginStream(iID);
    }
  }
  template <typename T>
  void ModuleHolderT<T>::endStream(StreamID iID) {
    if constexpr (HasStreamFunctions<T>) {
      m_mod->doEndStream(iID);
    }
  }

  namespace {
    template <typename T>
    concept HasRespondToInputFileFunctions = requires(T mod, FileBlock fb) {
      { mod.doRespondToOpenInputFile(fb) } -> std::same_as<void>;
      { mod.doRespondToCloseInputFile(fb) } -> std::same_as<void>;
    };

    template <typename T>
    concept HasRespondToCloseOutputFileFunction = requires(T mod) {
      { mod.doRespondToCloseOutputFile() } -> std::same_as<void>;
    };
  }  // namespace

  template <typename T>
  inline void ModuleHolderT<T>::implRespondToOpenInputFile(FileBlock const& fb) {
    if constexpr (HasRespondToInputFileFunctions<T>) {
      m_mod->doRespondToOpenInputFile(fb);
    }
  }

  template <typename T>
  inline void ModuleHolderT<T>::implRespondToCloseInputFile(FileBlock const& fb) {
    if constexpr (HasRespondToInputFileFunctions<T>) {
      m_mod->doRespondToCloseInputFile(fb);
    }
  }

  template <typename T>
  void ModuleHolderT<T>::implRespondToCloseOutputFile() {
    if constexpr (HasRespondToCloseOutputFileFunction<T>) {
      m_mod->doRespondToCloseOutputFile();
    }
  }

  namespace {
    template <typename T>
    bool mustPrefetchMayGet();

    template <>
    bool mustPrefetchMayGet<edm::one::EDProducerBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::one::EDFilterBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::one::EDAnalyzerBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::one::OutputModuleBase>() {
      return true;
    }

    template <>
    bool mustPrefetchMayGet<edm::global::EDProducerBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::global::EDFilterBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::global::EDAnalyzerBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::global::OutputModuleBase>() {
      return true;
    }

    template <>
    bool mustPrefetchMayGet<edm::limited::EDProducerBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::limited::EDFilterBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::limited::EDAnalyzerBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::limited::OutputModuleBase>() {
      return true;
    }

    template <>
    bool mustPrefetchMayGet<edm::stream::EDProducerAdaptorBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::stream::EDFilterAdaptorBase>() {
      return true;
    }
    template <>
    bool mustPrefetchMayGet<edm::stream::EDAnalyzerAdaptorBase>() {
      return true;
    }

  }  // namespace

  template <typename T>
  void ModuleHolderT<T>::updateLookup(BranchType iBranchType, ProductResolverIndexHelper const& iHelper) {
    m_mod->updateLookup(iBranchType, iHelper, mustPrefetchMayGet<T>());
  }

  template <typename T>
  void ModuleHolderT<T>::updateLookup(eventsetup::ESRecordsToProductResolverIndices const& iPI) {
    m_mod->updateLookup(iPI);
  }

  template <typename T>
  void ModuleHolderT<T>::releaseMemoryPostLookupSignal() {
    m_mod->releaseMemoryPostLookupSignal();
  }

  template <typename T>
  void ModuleHolderT<T>::selectInputProcessBlocks(ProductRegistry const& productRegistry,
                                                  ProcessBlockHelperBase const& processBlockHelperBase) {
    m_mod->selectInputProcessBlocks(productRegistry, processBlockHelperBase);
  }

  namespace {
    using ModuleToResolverIndicies =
        std::unordered_multimap<std::string, std::tuple<edm::TypeID const*, const char*, edm::ProductResolverIndex>>;
    void resolvePutIndiciesImpl(void*,
                                BranchType iBranchType,
                                ModuleToResolverIndicies const& iIndicies,
                                std::string const& iModuleLabel) {
      //Do nothing
    }

    void resolvePutIndiciesImpl(ProducerBase* iProd,
                                BranchType iBranchType,
                                ModuleToResolverIndicies const& iIndicies,
                                std::string const& iModuleLabel) {
      iProd->resolvePutIndicies(iBranchType, iIndicies, iModuleLabel);
    }

    void resolvePutIndiciesImpl(edm::stream::EDProducerAdaptorBase* iProd,
                                BranchType iBranchType,
                                ModuleToResolverIndicies const& iIndicies,
                                std::string const& iModuleLabel) {
      iProd->resolvePutIndicies(iBranchType, iIndicies, iModuleLabel);
    }
    void resolvePutIndiciesImpl(edm::stream::EDFilterAdaptorBase* iProd,
                                BranchType iBranchType,
                                ModuleToResolverIndicies const& iIndicies,
                                std::string const& iModuleLabel) {
      iProd->resolvePutIndicies(iBranchType, iIndicies, iModuleLabel);
    }

  }  // namespace

  template <typename T>
  void ModuleHolderT<T>::resolvePutIndicies(
      BranchType iBranchType,
      std::unordered_multimap<std::string, std::tuple<TypeID const*, const char*, edm::ProductResolverIndex>> const&
          iIndicies) {
    resolvePutIndiciesImpl(m_mod.get(), iBranchType, iIndicies, moduleDescription().moduleLabel());
  }

  template <typename T>
  std::vector<ModuleConsumesInfo> ModuleHolderT<T>::moduleConsumesInfos() const {
    return m_mod->moduleConsumesInfos();
  }
  template <typename T>
  std::vector<ModuleConsumesMinimalESInfo> ModuleHolderT<T>::moduleConsumesMinimalESInfos() const {
    return m_mod->moduleConsumesMinimalESInfos();
  }

  template <>
  ModuleHolder::Type ModuleHolderT<edm::one::EDProducerBase>::moduleType() const {
    return Type::kProducer;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::one::EDFilterBase>::moduleType() const {
    return Type::kFilter;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::one::EDAnalyzerBase>::moduleType() const {
    return Type::kAnalyzer;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::one::OutputModuleBase>::moduleType() const {
    return Type::kOutputModule;
  }

  template <>
  ModuleHolder::Type ModuleHolderT<edm::global::EDProducerBase>::moduleType() const {
    return Type::kProducer;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::global::EDFilterBase>::moduleType() const {
    return Type::kFilter;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::global::EDAnalyzerBase>::moduleType() const {
    return Type::kAnalyzer;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::global::OutputModuleBase>::moduleType() const {
    return Type::kOutputModule;
  }

  template <>
  ModuleHolder::Type ModuleHolderT<edm::limited::EDProducerBase>::moduleType() const {
    return Type::kProducer;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::limited::EDFilterBase>::moduleType() const {
    return Type::kFilter;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::limited::EDAnalyzerBase>::moduleType() const {
    return Type::kAnalyzer;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::limited::OutputModuleBase>::moduleType() const {
    return Type::kOutputModule;
  }

  template <>
  ModuleHolder::Type ModuleHolderT<edm::stream::EDProducerAdaptorBase>::moduleType() const {
    return Type::kProducer;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::stream::EDFilterAdaptorBase>::moduleType() const {
    return Type::kFilter;
  }
  template <>
  ModuleHolder::Type ModuleHolderT<edm::stream::EDAnalyzerAdaptorBase>::moduleType() const {
    return Type::kAnalyzer;
  }

  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::one::EDProducerBase>::moduleConcurrencyType() const {
    return Concurrency::kOne;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::one::EDFilterBase>::moduleConcurrencyType() const {
    return Concurrency::kOne;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::one::EDAnalyzerBase>::moduleConcurrencyType() const {
    return Concurrency::kOne;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::one::OutputModuleBase>::moduleConcurrencyType() const {
    return Concurrency::kOne;
  }

  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::global::EDProducerBase>::moduleConcurrencyType() const {
    return Concurrency::kGlobal;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::global::EDFilterBase>::moduleConcurrencyType() const {
    return Concurrency::kGlobal;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::global::EDAnalyzerBase>::moduleConcurrencyType() const {
    return Concurrency::kGlobal;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::global::OutputModuleBase>::moduleConcurrencyType() const {
    return Concurrency::kGlobal;
  }

  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::limited::EDProducerBase>::moduleConcurrencyType() const {
    return Concurrency::kLimited;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::limited::EDFilterBase>::moduleConcurrencyType() const {
    return Concurrency::kLimited;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::limited::EDAnalyzerBase>::moduleConcurrencyType() const {
    return Concurrency::kLimited;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::limited::OutputModuleBase>::moduleConcurrencyType() const {
    return Concurrency::kLimited;
  }

  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::stream::EDProducerAdaptorBase>::moduleConcurrencyType() const {
    return Concurrency::kStream;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::stream::EDFilterAdaptorBase>::moduleConcurrencyType() const {
    return Concurrency::kStream;
  }
  template <>
  ModuleHolder::Concurrency ModuleHolderT<edm::stream::EDAnalyzerAdaptorBase>::moduleConcurrencyType() const {
    return Concurrency::kStream;
  }

  //Explicitly instantiate our needed templates to avoid having the compiler
  // instantiate them in all of our libraries
  template class ModuleHolderT<one::EDProducerBase>;
  template class ModuleHolderT<one::EDFilterBase>;
  template class ModuleHolderT<one::EDAnalyzerBase>;
  template class ModuleHolderT<one::OutputModuleBase>;
  template class ModuleHolderT<global::EDProducerBase>;
  template class ModuleHolderT<global::EDFilterBase>;
  template class ModuleHolderT<global::EDAnalyzerBase>;
  template class ModuleHolderT<global::OutputModuleBase>;
  template class ModuleHolderT<stream::EDProducerAdaptorBase>;
  template class ModuleHolderT<stream::EDFilterAdaptorBase>;
  template class ModuleHolderT<stream::EDAnalyzerAdaptorBase>;
  template class ModuleHolderT<limited::EDProducerBase>;
  template class ModuleHolderT<limited::EDFilterBase>;
  template class ModuleHolderT<limited::EDAnalyzerBase>;
  template class ModuleHolderT<limited::OutputModuleBase>;

}  // namespace edm::maker
