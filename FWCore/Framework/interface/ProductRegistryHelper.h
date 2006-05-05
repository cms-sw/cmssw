#ifndef Framework_ProductRegistryHelper_h
#define Framework_ProductRegistryHelper_h

/*----------------------------------------------------------------------
  
ProductRegistryHelper: 

$Id: ProductRegistryHelper.h,v 1.5 2006/04/22 14:46:33 wmtan Exp $


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
  template <typename T> class Wrapper;
  class ProductRegistryHelper {
  public:

    ProductRegistryHelper() : typeLabelList_() {}
    virtual ~ProductRegistryHelper();
 
    struct TypeLabelItem {
      TypeLabelItem (TypeID const& tid, std::string const& pin, EDProduct const* edp) :
        typeID_(tid),
        productInstanceName_(pin),
#if 0
        branchAlias_(),
#endif
        productPtr_(edp) {}
      TypeID typeID_;
      std::string productInstanceName_;
#if 0
      std::string mutable branchAlias_;
      void setBranchAlias(std::string const& alias) const {branchAlias_ = alias;}
#endif
      boost::shared_ptr<EDProduct const> productPtr_; // pointer to a default constructed Wrapper<T>.
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
#if 0
    TypeLabelItem const& produces(std::string const& instanceName) {
#else
    void produces(std::string const& instanceName) {
#endif
      ProductType aproduct;
      TypeID tid(aproduct);
      TypeLabelItem tli(tid, instanceName, new Wrapper<ProductType>);
      typeLabelList_.push_back(tli);
#if 0
      return *typeLabelList_.rbegin();
#endif
    }

    template <class ProductType> 
#if 0
    TypeLabelItem const& produces() {
#else
    void produces() {
#endif
#if 0
      return produces<ProductType>(std::string());
#else
      produces<ProductType>(std::string());
#endif
    }

  private:
    TypeLabelList typeLabelList_;
  };


}

#endif
