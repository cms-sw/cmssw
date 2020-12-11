#ifndef FWCore_Framework_EDConsumerBase_h
#define FWCore_Framework_EDConsumerBase_h
// -*- C++ -*-
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
#include <map>
#include <string>
#include <vector>
#include <array>
#include <cassert>

// user include files
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/ProductResolverIndexAndSkipBit.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/HCTypeTag.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
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
  class ModuleDescription;
  class ModuleProcessName;
  class ProductResolverIndexHelper;
  class ProductRegistry;
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
    class ESRecordsToProxyIndices;
  }

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

    void itemsToGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const;
    void itemsMayGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const;

    std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFrom(BranchType iType) const {
      return itemsToGetFromBranch_[iType];
    }

    ///\return true if the product corresponding to the index was registered via consumes or mayConsume call
    bool registeredToConsume(ProductResolverIndex, bool, BranchType) const;

    ///\return true of TypeID corresponds to a type specified in a consumesMany call
    bool registeredToConsumeMany(TypeID const&, BranchType) const;
    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void updateLookup(BranchType iBranchType, ProductResolverIndexHelper const&, bool iPrefetchMayGet);
    void updateLookup(eventsetup::ESRecordsToProxyIndices const&);

    typedef ProductLabels Labels;
    void labelsForToken(EDGetToken iToken, Labels& oLabels) const;

    void modulesWhoseProductsAreConsumed(std::array<std::vector<ModuleDescription const*>*, NumBranchTypes>& modulesAll,
                                         std::vector<ModuleProcessName>& modulesInPreviousProcesses,
                                         ProductRegistry const& preg,
                                         std::map<std::string, ModuleDescription const*> const& labelsToDesc,
                                         std::string const& processName) const;

    /// Convert "@currentProcess" in InputTag process names to the actual current process name.
    void convertCurrentProcessAlias(std::string const& processName);

    std::vector<ConsumesInfo> consumesInfo() const;

    ESProxyIndex const* esGetTokenIndices(edm::Transition iTrans) const {
      if (iTrans < edm::Transition::NumberOfEventSetupTransitions) {
        auto const& v = esItemsToGetFromTransition_[static_cast<unsigned int>(iTrans)];
        if (v.empty()) {
          return nullptr;
        }
        return &(esItemsToGetFromTransition_[static_cast<unsigned int>(iTrans)].front());
      }
      return nullptr;
    }

    std::vector<ESProxyIndex> const& esGetTokenIndicesVector(edm::Transition iTrans) const {
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

    template <typename ProductType, BranchType B = InEvent>
    void consumesMany() {
      TypeToGet tid = TypeToGet::make<ProductType>();
      consumesMany<B>(tid);
    }

    void consumesMany(const TypeToGet& id) { consumesMany<InEvent>(id); }

    template <BranchType B>
    void consumesMany(const TypeToGet& id) {
      recordConsumes(B, id, edm::InputTag{}, true);
    }

    // For consuming event-setup products
    template <typename ESProduct, typename ESRecord, Transition Tr = Transition::Event>
    auto esConsumes() {
      return esConsumes<ESProduct, ESRecord, Tr>(ESInputTag{});
    }

    template <typename ESProduct, typename ESRecord, Transition Tr = Transition::Event>
    auto esConsumes(ESInputTag const& tag) {
      auto index = recordESConsumes(Tr,
                                    eventsetup::EventSetupRecordKey::makeKey<
                                        std::conditional_t<std::is_same_v<ESRecord, edm::DefaultRecord>,
                                                           eventsetup::default_record_t<ESHandleAdapter<ESProduct>>,
                                                           ESRecord>>(),
                                    eventsetup::heterocontainer::HCTypeTag::make<ESProduct>(),
                                    tag);
      return ESGetToken<ESProduct, ESRecord>{static_cast<unsigned int>(Tr), index, labelFor(index)};
    }

    template <Transition Tr = Transition::Event>
    [[nodiscard]] constexpr auto esConsumes() noexcept {
      return EDConsumerBaseESAdaptor<Tr>(this);
    }

    template <Transition Tr = Transition::Event>
    [[nodiscard]] auto esConsumes(ESInputTag tag) noexcept {
      return EDConsumerBaseWithTagESAdaptor<Tr>(this, std::move(tag));
    }

    ///Used with EventSetupRecord::doGet
    template <Transition Tr = Transition::Event>
    ESGetTokenGeneric esConsumes(eventsetup::EventSetupRecordKey const& iRecord, eventsetup::DataKey const& iKey) {
      return ESGetTokenGeneric(static_cast<unsigned int>(Tr),
                               recordESConsumes(Tr, iRecord, iKey.type(), ESInputTag("", iKey.name().value())),
                               iRecord.type());
    }

  private:
    virtual void registerLateConsumes(eventsetup::ESRecordsToProxyIndices const&) {}
    unsigned int recordConsumes(BranchType iBranch, TypeToGet const& iType, edm::InputTag const& iTag, bool iAlwaysGets);
    ESTokenIndex recordESConsumes(Transition,
                                  eventsetup::EventSetupRecordKey const&,
                                  eventsetup::heterocontainer::HCTypeTag const&,
                                  edm::ESInputTag const& iTag);

    const char* labelFor(ESTokenIndex) const;

    void throwTypeMismatch(edm::TypeID const&, EDGetToken) const;
    void throwBranchMismatch(BranchType, EDGetToken) const;
    void throwBadToken(edm::TypeID const& iType, EDGetToken iToken) const;
    void throwConsumesCallAfterFrozen(TypeToGet const&, InputTag const&) const;
    void throwESConsumesCallAfterFrozen(eventsetup::EventSetupRecordKey const&,
                                        eventsetup::heterocontainer::HCTypeTag const&,
                                        edm::ESInputTag const&) const;
    void throwESConsumesInProcessBlock() const;

    edm::InputTag const& checkIfEmpty(edm::InputTag const& tag);
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

    struct ESTokenLookupInfo {
      eventsetup::EventSetupRecordKey m_record;
      eventsetup::DataKey m_key;
      unsigned int m_startOfComponentName;
    };

    // TODO We would like to be able to access m_esTokenInfo from the
    // index in the token, but this is currently not possible. One idea
    // for this is to order the entries in m_esToken so that all the ones
    // for transition 0 come first, then the ones for for transition 1
    // and so on for all the transitions. Within a transition, the
    // entries would be in the same order in m_esTokenInfo and
    // esItemsToGetFromTransition_. This is something for future
    // development and might require a change to SoATuple to support
    // inserts in the middle of the data structure.
    enum { kESLookupInfo, kESProxyIndex };
    edm::SoATuple<ESTokenLookupInfo, ESProxyIndex> m_esTokenInfo;
    std::array<std::vector<ESProxyIndex>, static_cast<unsigned int>(edm::Transition::NumberOfEventSetupTransitions)>
        esItemsToGetFromTransition_;
    std::array<std::vector<ESRecordIndex>, static_cast<unsigned int>(edm::Transition::NumberOfEventSetupTransitions)>
        esRecordsToGetFromTransition_;
    bool frozen_;
    bool containsCurrentProcessAlias_;
  };

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
