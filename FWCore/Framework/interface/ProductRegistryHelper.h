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
  
  class ProductRegistryHelper {
  public:
    virtual ~ProductRegistryHelper() noexcept(false);
    ProductRegistryHelper() : typeLabelList_() {}

    // has_donotrecordparents<T>::value is true if we should not
    // record parentage for type T, and false otherwise.
    template <typename T>
    struct has_donotrecordparents {
      static constexpr bool value =
      std::is_base_of<DoNotRecordParents,T>::value;
    };

    struct TypeLabelItem {
      enum class AliasType { kBranchAlias, kSwitchAlias };

      TypeLabelItem (Transition const& transition, TypeID const& tid, std::string pin) :
	      transition_(transition),
        typeID_(tid),
        productInstanceName_(std::move(pin)),
        branchAlias_(),
        aliasType_(AliasType::kBranchAlias) {}
      Transition transition_;
      TypeID typeID_;
      std::string productInstanceName_;
      std::string branchAlias_;
      AliasType aliasType_;
    };

    struct BranchAliasSetter {
      BranchAliasSetter(TypeLabelItem& iItem, EDPutToken iToken):
      value_(iItem), token_(std::move(iToken)) {}
      
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
      
      operator EDPutToken() { return token_;}
    };

    template <typename T>
    struct BranchAliasSetterT {
      BranchAliasSetterT(TypeLabelItem& iItem, EDPutTokenT<T> iToken):
      value_(iItem), token_(std::move(iToken)) {}

      BranchAliasSetterT( BranchAliasSetter&& iS):
      value_(iS.value_), token_(iS.token_.index()) {}
      
      BranchAliasSetterT<T>& setBranchAlias(std::string alias) {
        value_.branchAlias_ = std::move(alias);
        return *this;
      }
      TypeLabelItem& value_;
      EDPutTokenT<T> token_;
      
      operator EDPutTokenT<T>() { return token_;}
      operator EDPutToken() { return EDPutToken(token_.index()); }
    };

    typedef std::vector<TypeLabelItem> TypeLabelList;

    /// used by the fwk to register the list of products of this module 
    TypeLabelList const& typeLabelList() const;
    
    std::vector<bool> const& recordProvenanceList() const { return recordProvenanceList_;}

    static
    void addToRegistry(TypeLabelList::const_iterator const& iBegin,
                             TypeLabelList::const_iterator const& iEnd,
                             ModuleDescription const& iDesc,
                             ProductRegistry& iReg,
                             ProductRegistryHelper* iProd,
                             bool iIsListener=false);

    /// declare what type of product will make and with which optional label 
    /** the statement
        \code
           produces<ProductType>("optlabel");
        \endcode
        should be added to the producer ctor for every product */


    template <class ProductType> 
    BranchAliasSetterT<ProductType> produces() {
      return produces<ProductType, InEvent>(std::string());
    }

    template <class ProductType> 
    BranchAliasSetterT<ProductType> produces(std::string instanceName) {
      return produces<ProductType, InEvent>( std::move(instanceName));
    }

    template <typename ProductType, BranchType B> 
    BranchAliasSetterT<ProductType> produces() {
      return produces<ProductType, B>(std::string());
    }

    template <typename ProductType, BranchType B> 
    BranchAliasSetterT<ProductType> produces(std::string instanceName) {
      TypeID tid(typeid(ProductType));
      return BranchAliasSetterT<ProductType>{produces<B>(tid,std::move(instanceName),
                                                         (not has_donotrecordparents<ProductType>::value) and B == InEvent)};
    }

    template <typename ProductType, Transition B>
    BranchAliasSetterT<ProductType> produces() {
      return produces<ProductType, B>(std::string());
    }
    
    template <typename ProductType, Transition B>
    BranchAliasSetterT<ProductType> produces(std::string instanceName) {
      TypeID tid(typeid(ProductType));
      return BranchAliasSetterT<ProductType>{produces<B>(tid,std::move(instanceName),
                                                         (not has_donotrecordparents<ProductType>::value) and B == Transition::Event)};
    }

   
    BranchAliasSetter produces(const TypeID& id, std::string instanceName=std::string(), bool recordProvenance = true) {
      return produces<Transition::Event>(id,std::move(instanceName),recordProvenance);
    }

    template <BranchType B>
    BranchAliasSetter produces(const TypeID& id, std::string instanceName=std::string(), bool recordProvenance = true) {
      unsigned int index =typeLabelList_.size();
      typeLabelList_.emplace_back(convertToTransition(B), id, std::move(instanceName));
      recordProvenanceList_.push_back(recordProvenance and B == InEvent);
      return BranchAliasSetter{typeLabelList_.back(),EDPutToken{static_cast<unsigned int>(index)}};
    }
    template <Transition B>
    BranchAliasSetter produces(const TypeID& id, std::string instanceName=std::string(), bool recordProvenance = true) {
      unsigned int index =typeLabelList_.size();
      typeLabelList_.emplace_back(B, id, std::move(instanceName));
      recordProvenanceList_.push_back(recordProvenance and B == Transition::Event);
      return BranchAliasSetter{typeLabelList_.back(),EDPutToken{ index }};
    }

    virtual bool hasAbilityToProduceInRuns() const {
      return false;
    }

    virtual bool hasAbilityToProduceInLumis() const {
      return false;
    }

  private:
    TypeLabelList typeLabelList_;
    std::vector<bool> recordProvenanceList_;
  };


}

#endif
