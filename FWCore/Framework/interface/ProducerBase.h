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
  
  class EDProducer;
  class EDFilter;
  namespace one {
    class EDProducerBase;
    class EDFilterBase;
  }
  namespace global {
    class EDProducerBase;
    class EDFilterBase;
  }
  namespace stream {
    class EDProducerWrapperBase;
    class EDFilterWrapperBase;
  }
  
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
    friend class EDProducer;
    friend class EDFilter;
    friend class one::EDProducerBase;
    friend class one::EDFilterBase;
    friend class global::EDProducerBase;
    friend class global::EDFilterBase;
    friend class stream::EDProducerWrapperBase;
    friend class stream::EDFilterWrapperBase;
    
    template< typename P>
    void commit_(P& iPrincipal) {
      iPrincipal.commit_();
    }

    template< typename P, typename L, typename I>
    void commit_(P& iPrincipal, L* iList, I* iID) {
      iPrincipal.commit_(iList,iID);
    }

    std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;
  };
}
#endif
