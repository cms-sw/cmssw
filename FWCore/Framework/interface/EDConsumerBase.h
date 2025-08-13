// -*- C++ -*-
#ifndef FWCore_Framework_EDConsumerBase_h
#define FWCore_Framework_EDConsumerBase_h
//
// Package:     FWCore/Framework
// Class  :     EDConsumerBase
//
/**\class edm::EDConsumerBase

 Description: Allows declaration of what data is being consumed

 Usage:
    The EDM modules all inherit from this base class

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 02 Apr 2013 21:35:53 GMT
//

// system include files
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <cassert>
#include <tuple>
#include <utility>

// user include files
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Framework/interface/ProductResolverIndexAndSkipBit.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/HCTypeTag.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/Framework/interface/ModuleConsumesMinimalESInfo.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/ServiceRegistry/interface/ModuleConsumesInfo.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ESIndices.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeToGet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESGetTokenGeneric.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/SoATuple.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/ProductLabels.h"

// forward declarations

namespace edm {
  class ProductResolverIndexHelper;
  class ConsumesCollector;
  template <Transition Tr>
  class EDConsumerBaseESAdaptor;
  template <Transition Tr>
  class EDConsumerBaseWithTagESAdaptor;
  template <BranchType B>
  class EDConsumerBaseAdaptor;
  template <typename T>
  class WillGetIfMatch;

  namespace eventsetup {
    struct ComponentDescription;
    class ESRecordsToProductResolverIndices;
  }  // namespace eventsetup

  class EDConsumerBase {
  public:
    EDConsumerBase();
    virtual ~EDConsumerBase() noexcept(false);

    // disallow copying
    EDConsumerBase(EDConsumerBase const&) = delete;
    EDConsumerBase const& operator=(EDConsumerBase const&) = delete;

    // allow moving
    EDConsumerBase(EDConsumerBase&&) = default;
    EDConsumerBase& operator=(EDConsumerBase&&) = default;

    // ---------- const member functions ---------------------
    ProductResolverIndexAndSkipBit indexFrom(EDGetToken, BranchType, TypeID const&) const;
    ProductResolverIndexAndSkipBit uncheckedIndexFrom(EDGetToken) const;
    /**returns edm::ProductResolverIndexInvalid if type and branch do not match the index
     this can happen if the same InputTag is used for different products */
    ProductResolverIndexAndSkipBit indexFromIfExactMatch(EDGetToken, BranchType, TypeID const&) const;
    EDGetToken getRegisteredToken(TypeID const& typeID,
                                  std::string const& label,
                                  std::string const& instance,
                                  std::string const& processName,
                                  BranchType branchType,
                                  bool skipCurrentProcess) const;

    void itemsToGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const;
    void itemsMayGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const;

    //used for prefetching
    std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFrom(BranchType iType) const {
      return itemsToGetFromBranch_[iType];
    }

    ///\return true if the product corresponding to the index was registered via consumes or mayConsume call
    bool registeredToConsume(ProductResolverIndex, bool, BranchType) const;

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void updateLookup(BranchType iBranchType, ProductResolverIndexHelper const&, bool iPrefetchMayGet);
    void updateLookup(eventsetup::ESRecordsToProductResolverIndices const&);
    void releaseMemoryPostLookupSignal();
    void selectInputProcessBlocks(ProductRegistry const& productRegistry,
                                  ProcessBlockHelperBase const& processBlockHelperBase) {
      doSelectInputProcessBlocks(productRegistry, processBlockHelperBase);
    }

    typedef ProductLabels Labels;
    void labelsForToken(EDGetToken iToken, Labels& oLabels) const;

    /// Convert "@currentProcess" in InputTag process names to the actual current process name.
    void convertCurrentProcessAlias(std::string const& processName);

    std::vector<ModuleConsumesInfo> moduleConsumesInfos() const;
    ///This can only be called before the end of beginJob (after that the underlying data has been deleted)
    /// The pointers held by ModuleConsumesMinimalESInfo will also be invalid at that time so copies of
    /// that class should not be held beyond the end of beginJob.
    std::vector<ModuleConsumesMinimalESInfo> moduleConsumesMinimalESInfos() const;

    ESResolverIndex const* esGetTokenIndices(edm::Transition iTrans) const {
      if (iTrans < edm::Transition::NumberOfEventSetupTransitions) {
        auto const& v = esItemsToGetFromTransition_[static_cast<unsigned int>(iTrans)];
        if (v.empty()) {
          return nullptr;
        }
        return &(esItemsToGetFromTransition_[static_cast<unsigned int>(iTrans)].front());
      }
      return nullptr;
    }

    std::vector<ESResolverIndex> const& esGetTokenIndicesVector(edm::Transition iTrans) const {
      assert(iTrans < edm::Transition::NumberOfEventSetupTransitions);
      return esItemsToGetFromTransition_[static_cast<unsigned int>(iTrans)];
    }

    std::vector<ESRecordIndex> const& esGetTokenRecordIndicesVector(edm::Transition iTrans) const {
      assert(iTrans < edm::Transition::NumberOfEventSetupTransitions);
      return esRecordsToGetFromTransition_[static_cast<unsigned int>(iTrans)];
    }

  protected:
    friend class ConsumesCollector;
    template <Transition Tr>
    friend class EDConsumerBaseESAdaptor;
    template <Transition Tr>
    friend class EDConsumerBaseWithTagESAdaptor;
    template <BranchType B>
    friend class EDConsumerBaseAdaptor;
    template <typename T>
    friend class WillGetIfMatch;
    ///Use a ConsumesCollector to gather consumes information from helper functions
    ConsumesCollector consumesCollector();

    template <typename ProductType, BranchType B = InEvent>
    EDGetTokenT<ProductType> consumes(edm::InputTag const& tag) {
      TypeToGet tid = TypeToGet::make<ProductType>();
      return EDGetTokenT<ProductType>{recordConsumes(B, tid, checkIfEmpty(tag), true)};
    }

    template <BranchType B = InEvent>
    [[nodiscard]] EDConsumerBaseAdaptor<B> consumes(edm::InputTag tag) noexcept {
      return EDConsumerBaseAdaptor<B>(this, std::move(tag));
    }

    EDGetToken consumes(const TypeToGet& id, edm::InputTag const& tag) {
      return EDGetToken{recordConsumes(InEvent, id, checkIfEmpty(tag), true)};
    }

    template <BranchType B>
    EDGetToken consumes(TypeToGet const& id, edm::InputTag const& tag) {
      return EDGetToken{recordConsumes(B, id, checkIfEmpty(tag), true)};
    }

    template <typename ProductType, BranchType B = InEvent>
    EDGetTokenT<ProductType> mayConsume(edm::InputTag const& tag) {
      TypeToGet tid = TypeToGet::make<ProductType>();
      return EDGetTokenT<ProductType>{recordConsumes(B, tid, checkIfEmpty(tag), false)};
    }

    EDGetToken mayConsume(const TypeToGet& id, edm::InputTag const& tag) { return mayConsume<InEvent>(id, tag); }

    template <BranchType B>
    EDGetToken mayConsume(const TypeToGet& id, edm::InputTag const& tag) {
      return EDGetToken{recordConsumes(B, id, checkIfEmpty(tag), false)};
    }

    // For consuming event-setup products
    template <typename ESProduct, typename ESRecord, Transition Tr = Transition::Event>
    auto esConsumes() {
      return esConsumes<ESProduct, ESRecord, Tr>(ESInputTag{});
    }

    template <typename ESProduct, typename ESRecord, Transition Tr = Transition::Event>
    auto esConsumes(ESInputTag const& tag) {
      auto [index, productLabel] =
          recordESConsumes(Tr,
                           eventsetup::EventSetupRecordKey::makeKey<
                               std::conditional_t<std::is_same_v<ESRecord, edm::DefaultRecord>,
                                                  eventsetup::default_record_t<ESHandleAdapter<ESProduct>>,
                                                  ESRecord>>(),
                           eventsetup::heterocontainer::HCTypeTag::make<ESProduct>(),
                           tag);
      return ESGetToken<ESProduct, ESRecord>{static_cast<unsigned int>(Tr), index, productLabel};
    }

    template <Transition Tr = Transition::Event>
    [[nodiscard]] constexpr auto esConsumes() {
      return EDConsumerBaseESAdaptor<Tr>(this);
    }

    template <Transition Tr = Transition::Event>
    [[nodiscard]] auto esConsumes(ESInputTag tag) {
      return EDConsumerBaseWithTagESAdaptor<Tr>(this, std::move(tag));
    }

    ///Used with EventSetupRecord::doGet
    template <Transition Tr = Transition::Event>
    ESGetTokenGeneric esConsumes(eventsetup::EventSetupRecordKey const& iRecord, eventsetup::DataKey const& iKey) {
      auto [index, productLabel] = recordESConsumes(Tr, iRecord, iKey.type(), ESInputTag("", iKey.name().value()));
      return ESGetTokenGeneric(static_cast<unsigned int>(Tr), index, iRecord.type());
    }

    /**The passed functor must take the following signature
     *  F( ModuleConsumesInfo const& )
     * The functor will be called for each consumed EDProduct registered for the module
     */
    template <typename F>
    void consumedProducts(F&& iFunc) const;

    /**The passed functor must take the following signature
     * F(edm::ModuleConsumesMinimalESInfo const& )
     * The functor will be called for each consumed ESProduct registered for the module
     */
    template <typename F>
    void consumedESProducts(F&& iFunct) const;

  private:
    virtual void extendUpdateLookup(BranchType iBranchType, ProductResolverIndexHelper const&);
    virtual void registerLateConsumes(eventsetup::ESRecordsToProductResolverIndices const&) {}
    unsigned int recordConsumes(BranchType iBranch, TypeToGet const& iType, edm::InputTag const& iTag, bool iAlwaysGets);
    std::tuple<ESTokenIndex, char const*> recordESConsumes(Transition,
                                                           eventsetup::EventSetupRecordKey const&,
                                                           eventsetup::heterocontainer::HCTypeTag const&,
                                                           edm::ESInputTag const& iTag);

    void throwTypeMismatch(edm::TypeID const&, EDGetToken) const;
    void throwBranchMismatch(BranchType, EDGetToken) const;
    void throwBadToken(edm::TypeID const& iType, EDGetToken iToken) const;
    void throwConsumesCallAfterFrozen(TypeToGet const&, InputTag const&) const;
    void throwESConsumesCallAfterFrozen(eventsetup::EventSetupRecordKey const&,
                                        eventsetup::heterocontainer::HCTypeTag const&,
                                        edm::ESInputTag const&) const;
    void throwESConsumesInProcessBlock() const;

    edm::InputTag const& checkIfEmpty(edm::InputTag const& tag);

    virtual void doSelectInputProcessBlocks(ProductRegistry const&, ProcessBlockHelperBase const&);

    struct ESTokenLookupInfo {
      eventsetup::EventSetupRecordKey m_record;
      eventsetup::DataKey m_key;
      unsigned int m_startOfComponentName;
    };

    enum { kESLookupInfo, kESResolverIndex };

    using ESTokenLookupInfoContainer = edm::SoATuple<ESTokenLookupInfo, ESResolverIndex>;

    ESTokenLookupInfoContainer const& esTokenLookupInfoContainer() const {
      return esDataThatCanBeDeletedEarly_->esTokenLookupInfoContainer_;
    }

    using ESResolverIndexContainer = std::array<std::vector<ESResolverIndex>, kNumberOfEventSetupTransitions>;

    using ConsumesIndexConverter =
        std::vector<std::pair<ESResolverIndexContainer::size_type, std::vector<ESResolverIndex>::size_type>>;

    // This can be used to convert from an index used to access esTokenLookupInfoContainer_
    // into the 2 indexes needed to access esItemsToGetFromTransition_
    ConsumesIndexConverter const& consumesIndexConverter() const {
      return esDataThatCanBeDeletedEarly_->consumesIndexConverter_;
    }

    // ---------- member data --------------------------------

    struct TokenLookupInfo {
      TokenLookupInfo(edm::TypeID const& iID, ProductResolverIndex iIndex, bool skipCurrentProcess, BranchType iBranch)
          : m_type(iID), m_index(iIndex, skipCurrentProcess), m_branchType(iBranch) {}
      edm::TypeID m_type;
      ProductResolverIndexAndSkipBit m_index;
      BranchType m_branchType;
    };

    struct LabelPlacement {
      LabelPlacement(unsigned int iStartOfModuleLabel,
                     unsigned short iDeltaToProductInstance,
                     unsigned short iDeltaToProcessName)
          : m_startOfModuleLabel(iStartOfModuleLabel),
            m_deltaToProductInstance(iDeltaToProductInstance),
            m_deltaToProcessName(iDeltaToProcessName) {}
      unsigned int m_startOfModuleLabel;
      unsigned short m_deltaToProductInstance;
      unsigned short m_deltaToProcessName;
    };

    //define the purpose of each 'column' in m_tokenInfo
    enum { kLookupInfo, kAlwaysGets, kLabels, kKind };
    edm::SoATuple<TokenLookupInfo, bool, LabelPlacement, edm::KindOfType> m_tokenInfo;

    //m_tokenStartOfLabels holds the entries into this container
    // for each of the 3 labels needed to id the data
    std::vector<char> m_tokenLabels;

    std::array<std::vector<ProductResolverIndexAndSkipBit>, edm::NumBranchTypes> itemsToGetFromBranch_;

    struct ESDataThatCanBeDeletedEarly {
      ESTokenLookupInfoContainer esTokenLookupInfoContainer_;
      ConsumesIndexConverter consumesIndexConverter_;
    };

    std::unique_ptr<ESDataThatCanBeDeletedEarly> esDataThatCanBeDeletedEarly_;

    ESResolverIndexContainer esItemsToGetFromTransition_;

    std::array<std::vector<ESRecordIndex>, kNumberOfEventSetupTransitions> esRecordsToGetFromTransition_;

    bool frozen_;
    bool containsCurrentProcessAlias_;
  };

  template <typename F>
  void EDConsumerBase::consumedProducts(F&& iFunc) const {
    auto itKind = m_tokenInfo.begin<kKind>();
    auto itLabels = m_tokenInfo.begin<kLabels>();
    auto itAlways = m_tokenInfo.begin<kAlwaysGets>();
    for (auto itInfo = m_tokenInfo.begin<kLookupInfo>(), itEnd = m_tokenInfo.end<kLookupInfo>(); itInfo != itEnd;
         ++itInfo, ++itKind, ++itLabels, ++itAlways) {
      auto labels = *itLabels;
      unsigned int start = labels.m_startOfModuleLabel;
      auto module = &(m_tokenLabels[start]);
      auto productInstance = module + labels.m_deltaToProductInstance;
      auto process = module + labels.m_deltaToProcessName;

      iFunc(ModuleConsumesInfo(itInfo->m_type,
                               module,
                               productInstance,
                               process,
                               itInfo->m_branchType,
                               *itKind,
                               *itAlways,
                               itInfo->m_index.skipCurrentProcess()));
    }
  }

  template <typename F>
  void EDConsumerBase::consumedESProducts(F&& iFunc) const {
    unsigned int index = 0;
    auto itResolverIndex = esTokenLookupInfoContainer().begin<kESResolverIndex>();
    for (auto it = esTokenLookupInfoContainer().begin<kESLookupInfo>();
         it != esTokenLookupInfoContainer().end<kESLookupInfo>();
         ++it, ++index, ++itResolverIndex) {
      //NOTE: memory for it->m_key is passed on to call via the ModuleConsumesMinimalESInfo constructor
      // this avoids a copy and avoids code later in the call chain accidently using deleted memory
      iFunc(ModuleConsumesMinimalESInfo(static_cast<Transition>(consumesIndexConverter()[index].first),
                                        it->m_record,
                                        it->m_key,
                                        &(m_tokenLabels[it->m_startOfComponentName]),
                                        *itResolverIndex));
    }
  }

  template <Transition TR>
  class EDConsumerBaseESAdaptor {
  public:
    template <typename TYPE, typename REC>
    ESGetToken<TYPE, REC> consumes() {
      return m_consumer->template esConsumes<TYPE, REC, TR>();
    }

  private:
    //only EDConsumerBase is allowed to make an instance of this class
    friend class EDConsumerBase;
    EDConsumerBaseESAdaptor(EDConsumerBase* iBase) : m_consumer(iBase) {}

    EDConsumerBase* m_consumer;
  };

  template <Transition TR>
  class EDConsumerBaseWithTagESAdaptor {
  public:
    template <typename TYPE, typename REC>
    ESGetToken<TYPE, REC> consumes() {
      return m_consumer->template esConsumes<TYPE, REC, TR>(m_tag);
    }

  private:
    //only EDConsumerBase is allowed to make an instance of this class
    friend class EDConsumerBase;
    EDConsumerBaseWithTagESAdaptor(EDConsumerBase* iBase, ESInputTag iTag) noexcept
        : m_consumer(iBase), m_tag(std::move(iTag)) {}

    EDConsumerBase* m_consumer;
    ESInputTag const m_tag;
  };

  template <BranchType B>
  class EDConsumerBaseAdaptor {
  public:
    template <typename TYPE>
    EDGetTokenT<TYPE> consumes() {
      return m_consumer->template consumes<TYPE, B>(m_tag);
    }

  private:
    //only EDConsumerBase is allowed to make an instance of this class
    friend class EDConsumerBase;
    EDConsumerBaseAdaptor(EDConsumerBase* iBase, edm::InputTag iTag) noexcept
        : m_consumer(iBase), m_tag(std::move(iTag)) {}

    EDConsumerBase* m_consumer;
    edm::InputTag const m_tag;
  };

}  // namespace edm

#endif
