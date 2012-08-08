#ifndef FWCore_Framework_ProducerBase_h
#define FWCore_Framework_ProducerBase_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"

#include <functional>

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
    std::function<void(BranchDescription const&)> registrationCallback() const;

    void registerProducts(ProducerBase*,
			ProductRegistry*,
			ModuleDescription const&);

    using ProductRegistryHelper::produces;
    using ProductRegistryHelper::typeLabelList;

  protected:
    void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func) {
       callWhenNewProductsRegistered_ = func;
    }
          
  private:
    std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;
  };
}
#endif
