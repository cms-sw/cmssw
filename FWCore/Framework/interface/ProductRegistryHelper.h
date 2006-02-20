#ifndef Framework_ProductRegistryHelper_h
#define Framework_ProductRegistryHelper_h

/*----------------------------------------------------------------------
  
ProductRegistryHelper: 

$Id: ProductRegistryHelper.h,v 1.1 2005/12/28 00:30:09 wmtan Exp $


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/TypeID.h"
#include <string>
#include <list>
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
      TypeLabelItem (TypeID const& tid, std::string const& pin, EDProduct * edp) :
        typeID_(tid), productInstanceName_(pin), productPtr_(edp) {}
      TypeID typeID_;
      std::string productInstanceName_;
      EDProduct *productPtr_; // pointer to a default constructed Wrapper<T>.
    };

    /// declare what type of product will make and with which optional label 
    /** the statement
        \code
           produces<ProductType>("optlabel");
        \endcode
        should be added to the producer ctor for every product */

    template <class ProductType> 
    void produces(std::string const& instanceName) {
      ProductType aproduct;
      TypeID tid(aproduct);
      TypeLabelItem tli(tid, instanceName, new Wrapper<ProductType>);
      typeLabelList_.push_back(tli);
    }

    template <class ProductType> 
    void produces(){
      produces<ProductType>(std::string());
    }

    typedef std::list<TypeLabelItem> TypeLabelList;

    /// used by the fwk to register the list of products of this module 
    TypeLabelList & typeLabelList();

    static
    void addToRegistry(TypeLabelList::const_iterator const& iBegin,
                             TypeLabelList::const_iterator const& iEnd,
                             ModuleDescription const& iDesc,
                             ProductRegistry& iReg,
                             bool iIsListener=false);

  private:
    TypeLabelList typeLabelList_;
  };


}

#endif
