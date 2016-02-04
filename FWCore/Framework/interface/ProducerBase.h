#ifndef FWCore_Framework_ProducerBase_h
#define FWCore_Framework_ProducerBase_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "boost/bind.hpp"
#include "boost/function.hpp"
namespace edm {
  class BranchDescription;
  class ModuleDescription;
  class ProductRegistry;
  class ProducerBase : private ProductRegistryHelper {
  public:
    typedef ProductRegistryHelper::TypeLabelList TypeLabelList;
    ProducerBase ();
    virtual ~ProducerBase();
 
    /// used by the fwk to register list of products
    boost::function<void(const BranchDescription&)> registrationCallback() const;

    void registerProducts(ProducerBase*,
			ProductRegistry*,
			ModuleDescription const&);

    using ProductRegistryHelper::produces;
    using ProductRegistryHelper::typeLabelList;

  protected:
    template<class TProducer, class TMethod>
    void callWhenNewProductsRegistered(TProducer* iProd, TMethod iMethod) {
       callWhenNewProductsRegistered_ = boost::bind(iMethod,iProd,_1);
    }
          
  private:
    boost::function<void(const BranchDescription&)> callWhenNewProductsRegistered_;
  };


}

#endif
