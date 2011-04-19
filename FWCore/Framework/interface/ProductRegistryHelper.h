#ifndef FWCore_Framework_ProductRegistryHelper_h
#define FWCore_Framework_ProductRegistryHelper_h

/*----------------------------------------------------------------------
  
ProductRegistryHelper: 

$Id: ProductRegistryHelper.h,v 1.14 2008/02/12 21:46:26 chrjones Exp $


----------------------------------------------------------------------*/

#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include <string>
#include <list>

namespace edm {
  class WrapperHolder;
  class ModuleDescription;
  class ProductRegistry;
  class ProductRegistryHelper {
  public:

    ProductRegistryHelper() : typeLabelList_() {}
    ~ProductRegistryHelper();
 
    struct TypeLabelItem {
      TypeLabelItem (BranchType const& branchType, TypeID const& tid, std::string const& pin) :
	branchType_(branchType),
        typeID_(tid),
        productInstanceName_(pin),
        branchAlias_() {}
      BranchType branchType_;
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

   
    TypeLabelItem const& produces(const TypeID& id, std::string const& instanceName=std::string()) {
       return produces<InEvent>(id,instanceName);
    }

    template <BranchType B>
    TypeLabelItem const& produces(const TypeID& id, std::string const& instanceName=std::string()) {
       TypeLabelItem tli(B, id, instanceName);
       typeLabelList_.push_back(tli);
       return *typeLabelList_.rbegin();
    }
  private:
    TypeLabelList typeLabelList_;
  };


}

#endif
