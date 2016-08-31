#ifndef FWCore_Framework_ProducerBase_h
#define FWCore_Framework_ProducerBase_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"

#include <functional>
#include <unordered_map>
#include <string>
#include<vector>

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
    template<typename T> class ProducingModuleAdaptorBase;
  }
  
  class ProducerBase : private ProductRegistryHelper {
  public:
    typedef ProductRegistryHelper::TypeLabelList TypeLabelList;
    ProducerBase ();
    virtual ~ProducerBase() noexcept(false);
 
    /// used by the fwk to register list of products
    std::function<void(BranchDescription const&)> registrationCallback() const;

    void registerProducts(ProducerBase*,
			ProductRegistry*,
			ModuleDescription const&);

    using ProductRegistryHelper::produces;
    using ProductRegistryHelper::typeLabelList;

    void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func) {
       callWhenNewProductsRegistered_ = func;
    }
    
    void resolvePutIndicies(BranchType iBranchType,
                            std::unordered_multimap<std::string, edm::ProductResolverIndex> const& iIndicies,
                            std::string const& moduleLabel);
    
    std::vector<edm::ProductResolverIndex> const& indiciesForPutProducts(BranchType iBranchType) const {
      return putIndicies_[iBranchType];
    }
  private:
    friend class EDProducer;
    friend class EDFilter;
    friend class one::EDProducerBase;
    friend class one::EDFilterBase;
    friend class global::EDProducerBase;
    friend class global::EDFilterBase;
    template<typename T> friend class stream::ProducingModuleAdaptorBase;
    
    template< typename P>
    void commit_(P& iPrincipal) {
      iPrincipal.commit_();
    }

    template< typename P, typename L, typename I>
    void commit_(P& iPrincipal, L* iList, I* iID) {
      iPrincipal.commit_(iList,iID);
    }

    std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;
    std::array<std::vector<edm::ProductResolverIndex>, edm::NumBranchTypes> putIndicies_;
  };
}
#endif
