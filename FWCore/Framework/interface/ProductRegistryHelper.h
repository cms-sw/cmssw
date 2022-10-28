#ifndef FWCore_Framework_ProductRegistryHelper_h
#define FWCore_Framework_ProductRegistryHelper_h

/*----------------------------------------------------------------------

ProductRegistryHelper:

----------------------------------------------------------------------*/

#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include <string>
#include <vector>
#include <type_traits>

namespace edm {
  class ModuleDescription;
  class ProductRegistry;
  struct DoNotRecordParents;

  template <Transition B>
  class ProductRegistryHelperAdaptor;

  class ProductRegistryHelper {
  public:
    virtual ~ProductRegistryHelper() noexcept(false);
    ProductRegistryHelper() : typeLabelList_() {}

    // has_donotrecordparents<T>::value is true if we should not
    // record parentage for type T, and false otherwise.
    template <typename T>
    struct has_donotrecordparents {
      static constexpr bool value = std::is_base_of<DoNotRecordParents, T>::value;
    };

    struct TypeLabelItem {
      enum class AliasType : char { kBranchAlias, kSwitchAlias };

      TypeLabelItem(Transition const& transition, TypeID const& tid, std::string pin)
          : transition_(transition),
            typeID_(tid),
            productInstanceName_(std::move(pin)),
            branchAlias_(),
            aliasType_(AliasType::kBranchAlias),
            isTransform_(false) {}
      Transition transition_;
      TypeID typeID_;
      std::string productInstanceName_;
      std::string branchAlias_;
      AliasType aliasType_;
      bool isTransform_;
    };

    struct BranchAliasSetter {
      BranchAliasSetter(TypeLabelItem& iItem, EDPutToken iToken) : value_(iItem), token_(iToken) {}

      BranchAliasSetter& setBranchAlias(std::string alias) {
        value_.branchAlias_ = std::move(alias);
        return *this;
      }
      BranchAliasSetter& setSwitchAlias(std::string moduleLabel) {
        value_.branchAlias_ = std::move(moduleLabel);
        value_.aliasType_ = TypeLabelItem::AliasType::kSwitchAlias;
        return *this;
      }
      TypeLabelItem& value_;
      EDPutToken token_;

      operator EDPutToken() { return token_; }
    };

    template <typename T>
    struct BranchAliasSetterT {
      BranchAliasSetterT(TypeLabelItem& iItem, EDPutTokenT<T> iToken) : value_(iItem), token_(std::move(iToken)) {}

      BranchAliasSetterT(BranchAliasSetter&& iS) : value_(iS.value_), token_(iS.token_.index()) {}

      BranchAliasSetterT<T>& setBranchAlias(std::string alias) {
        value_.branchAlias_ = std::move(alias);
        return *this;
      }
      TypeLabelItem& value_;
      EDPutTokenT<T> token_;

      template <typename U>
      EDPutTokenT<T> produces() {
        static_assert(std::is_same_v<T, U>);
        return token_;
      }

      operator EDPutTokenT<T>() { return token_; }
      operator EDPutToken() { return EDPutToken(token_.index()); }
    };

    typedef std::vector<TypeLabelItem> TypeLabelList;

    /// used by the fwk to register the list of products of this module
    TypeLabelList const& typeLabelList() const;

    std::vector<bool> const& recordProvenanceList() const { return recordProvenanceList_; }

    static void addToRegistry(TypeLabelList::const_iterator const& iBegin,
                              TypeLabelList::const_iterator const& iEnd,
                              ModuleDescription const& iDesc,
                              ProductRegistry& iReg,
                              ProductRegistryHelper* iProd,
                              bool iIsListener = false);

    /// declare what type of product will make and with which optional label
    /** the statement
        \code
           produces<ProductType>("optlabel");
        \endcode
        should be added to the producer ctor for every product */

    template <Transition Tr = Transition::Event>
    [[nodiscard]] auto produces(std::string instanceName) noexcept {
      return ProductRegistryHelperAdaptor<Tr>(*this, std::move(instanceName));
    }
    template <Transition Tr = Transition::Event>
    [[nodiscard]] auto produces() noexcept {
      return ProductRegistryHelperAdaptor<Tr>(*this);
    }

    template <class ProductType>
    BranchAliasSetterT<ProductType> produces() {
      return produces<ProductType, InEvent>(std::string());
    }

    template <class ProductType>
    BranchAliasSetterT<ProductType> produces(std::string instanceName) {
      return produces<ProductType, InEvent>(std::move(instanceName));
    }

    template <typename ProductType, BranchType B>
    BranchAliasSetterT<ProductType> produces() {
      return produces<ProductType, B>(std::string());
    }

    template <typename ProductType, BranchType B>
    BranchAliasSetterT<ProductType> produces(std::string instanceName) {
      TypeID tid(typeid(ProductType));
      return BranchAliasSetterT<ProductType>{
          produces<B>(tid, std::move(instanceName), (not has_donotrecordparents<ProductType>::value) and B == InEvent)};
    }

    template <typename ProductType, Transition B>
    BranchAliasSetterT<ProductType> produces() {
      return produces<ProductType, B>(std::string());
    }

    template <typename ProductType, Transition B>
    BranchAliasSetterT<ProductType> produces(std::string instanceName) {
      TypeID tid(typeid(ProductType));
      return BranchAliasSetterT<ProductType>{produces<B>(
          tid, std::move(instanceName), (not has_donotrecordparents<ProductType>::value) and B == Transition::Event)};
    }

    BranchAliasSetter produces(const TypeID& id,
                               std::string instanceName = std::string(),
                               bool recordProvenance = true) {
      return produces<Transition::Event>(id, std::move(instanceName), recordProvenance);
    }

    template <BranchType B>
    BranchAliasSetter produces(const TypeID& id,
                               std::string instanceName = std::string(),
                               bool recordProvenance = true) {
      unsigned int index = typeLabelList_.size();
      typeLabelList_.emplace_back(convertToTransition(B), id, std::move(instanceName));
      recordProvenanceList_.push_back(recordProvenance and B == InEvent);
      return BranchAliasSetter{typeLabelList_.back(), EDPutToken{static_cast<unsigned int>(index)}};
    }
    template <Transition B>
    BranchAliasSetter produces(const TypeID& id,
                               std::string instanceName = std::string(),
                               bool recordProvenance = true) {
      unsigned int index = typeLabelList_.size();
      typeLabelList_.emplace_back(B, id, std::move(instanceName));
      recordProvenanceList_.push_back(recordProvenance and B == Transition::Event);
      return BranchAliasSetter{typeLabelList_.back(), EDPutToken{index}};
    }

    EDPutToken transforms(const TypeID& id, std::string instanceName) {
      unsigned int index = typeLabelList_.size();
      typeLabelList_.emplace_back(Transition::Event, id, std::move(instanceName));
      typeLabelList_.back().isTransform_ = true;
      recordProvenanceList_.push_back(true);
      return EDPutToken{index};
    }

    virtual bool hasAbilityToProduceInBeginProcessBlocks() const { return false; }
    virtual bool hasAbilityToProduceInEndProcessBlocks() const { return false; }

    virtual bool hasAbilityToProduceInBeginRuns() const { return false; }
    virtual bool hasAbilityToProduceInEndRuns() const { return false; }

    virtual bool hasAbilityToProduceInBeginLumis() const { return false; }
    virtual bool hasAbilityToProduceInEndLumis() const { return false; }

  private:
    TypeLabelList typeLabelList_;
    std::vector<bool> recordProvenanceList_;
  };

  template <Transition B>
  class ProductRegistryHelperAdaptor {
  public:
    template <typename TYPE>
    EDPutTokenT<TYPE> produces() {
      return m_helper.template produces<TYPE, B>(m_label);
    }

  private:
    //only ProductRegistryHelper is allowed to make an instance of this class
    friend class ProductRegistryHelper;

    ProductRegistryHelperAdaptor(ProductRegistryHelper& iBase, std::string iLabel)
        : m_helper(iBase), m_label(std::move(iLabel)) {}
    explicit ProductRegistryHelperAdaptor(ProductRegistryHelper& iBase) : m_helper(iBase), m_label() {}

    ProductRegistryHelper& m_helper;
    std::string const m_label;
  };

}  // namespace edm

#endif
