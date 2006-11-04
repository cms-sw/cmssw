#ifndef Framework_ProductRegistryHelper_h
#define Framework_ProductRegistryHelper_h

/*----------------------------------------------------------------------
  
ProductRegistryHelper: 

$Id: ProductRegistryHelper.h,v 1.9 2006/08/30 23:34:01 wmtan Exp $


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/TypeID.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/BranchType.h"
#include <string>
#include <list>
#include "boost/shared_ptr.hpp"

namespace edm {
  class EDProduct;
  class ModuleDescription;
  class ProductRegistry;
  class ProductRegistryHelper {
  public:

    ProductRegistryHelper() : typeLabelList_() {}
    virtual ~ProductRegistryHelper();
 
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

      ProductType aproduct;
      TypeID tid(aproduct);
      TypeLabelItem tli(B, tid, instanceName);
      typeLabelList_.push_back(tli);
      return *typeLabelList_.rbegin();
    }

  private:
    TypeLabelList typeLabelList_;
  };


}

#endif
