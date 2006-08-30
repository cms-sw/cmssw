#ifndef Framework_ProductRegistryHelper_h
#define Framework_ProductRegistryHelper_h

/*----------------------------------------------------------------------
  
ProductRegistryHelper: 

$Id: ProductRegistryHelper.h,v 1.8 2006/05/24 01:52:51 wmtan Exp $


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/TypeID.h"
#include "DataFormats/Common/interface/EDProduct.h"
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
      TypeLabelItem (TypeID const& tid, std::string const& pin) :
        typeID_(tid),
        productInstanceName_(pin),
        branchAlias_() {}
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
    TypeLabelItem const& produces(std::string const& instanceName) {

      ProductType aproduct;
      TypeID tid(aproduct);
      TypeLabelItem tli(tid, instanceName);
      typeLabelList_.push_back(tli);
      return *typeLabelList_.rbegin();
    }

    template <class ProductType> 
    TypeLabelItem const& produces() {
      return produces<ProductType>(std::string());
    }

  private:
    TypeLabelList typeLabelList_;
  };


}

#endif
