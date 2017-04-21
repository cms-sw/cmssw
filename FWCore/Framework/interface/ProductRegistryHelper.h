#ifndef FWCore_Framework_ProductRegistryHelper_h
#define FWCore_Framework_ProductRegistryHelper_h

/*----------------------------------------------------------------------
  
ProductRegistryHelper: 

----------------------------------------------------------------------*/

#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include <string>
#include <list>

namespace edm {
  class ModuleDescription;
  class ProductRegistry;
  class ProductRegistryHelper {
  public:

    ProductRegistryHelper() : typeLabelList_() {}
    ~ProductRegistryHelper();
 
    struct TypeLabelItem {
      TypeLabelItem (Transition const& transition, TypeID const& tid, std::string const& pin) :
	      transition_(transition),
        typeID_(tid),
        productInstanceName_(pin),
        branchAlias_() {}
      Transition transition_;
      TypeID typeID_;
      std::string productInstanceName_;
      mutable std::string branchAlias_;
      void setBranchAlias(std::string const& alias) const {branchAlias_ = alias;}
    };

    typedef std::list<TypeLabelItem> TypeLabelList;

    /// used by the fwk to register the list of products of this module 
    TypeLabelList & typeLabelList();

    static
    void addToRegistry(TypeLabelList::const_iterator const& iBegin,
                             TypeLabelList::const_iterator const& iEnd,
                             ModuleDescription const& iDesc,
                             ProductRegistry& iReg,
                             bool iIsListener=false);

    /// declare what type of product will make and with which optional label 
    /** the statement
        \code
           produces<ProductType>("optlabel");
        \endcode
        should be added to the producer ctor for every product */


    template <class ProductType> 
    TypeLabelItem const& produces() {
      return produces<ProductType, InEvent>(std::string());
    }

    template <class ProductType> 
    TypeLabelItem const& produces(std::string const& instanceName) {
      return produces<ProductType, InEvent>(instanceName);
    }

    template <typename ProductType, BranchType B> 
    TypeLabelItem const& produces() {
      return produces<ProductType, B>(std::string());
    }

    template <typename ProductType, BranchType B> 
    TypeLabelItem const& produces(std::string const& instanceName) {
      TypeID tid(typeid(ProductType));
      return produces<B>(tid,instanceName);
    }

    template <typename ProductType, Transition B>
    TypeLabelItem const& produces() {
      return produces<ProductType, B>(std::string());
    }
    
    template <typename ProductType, Transition B>
    TypeLabelItem const& produces(std::string const& instanceName) {
      TypeID tid(typeid(ProductType));
      return produces<B>(tid,instanceName);
    }

   
    TypeLabelItem const& produces(const TypeID& id, std::string const& instanceName=std::string()) {
      return produces<Transition::Event>(id,instanceName);
    }

    template <BranchType B>
    TypeLabelItem const& produces(const TypeID& id, std::string const& instanceName=std::string()) {
       typeLabelList_.emplace_back(convertToTransition(B), id, instanceName);
       return *typeLabelList_.rbegin();
    }
    template <Transition B>
    TypeLabelItem const& produces(const TypeID& id, std::string const& instanceName=std::string()) {
      typeLabelList_.emplace_back(B, id, instanceName);
      return *typeLabelList_.rbegin();
    }

  private:
    TypeLabelList typeLabelList_;
  };


}

#endif
