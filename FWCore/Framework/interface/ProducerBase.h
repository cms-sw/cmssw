#ifndef FWCore_Framework_ProducerBase_h
#define FWCore_Framework_ProducerBase_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

$Id: ProducerBase.h,v 1.3 2006/12/08 21:40:28 wmtan Exp $


----------------------------------------------------------------------*/

#include "FWCore/Utilities/interface/GCCPrerequisite.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "boost/bind.hpp"
#include "boost/function.hpp"
#include "boost/shared_ptr.hpp"
#include <string>
namespace edm {
  class BranchDescription;
  class ModuleDescription;
  class ProductRegistry;
#if GCC_PREREQUISITE(3,4,4)
  class ProducerBase : private ProductRegistryHelper {
#else
  // Bug in gcc3.2.3 compiler forces public inheritance
  class ProducerBase : public ProductRegistryHelper {
#endif
  public:
    typedef ProductRegistryHelper::TypeLabelList TypeLabelList;
    ProducerBase ();
    virtual ~ProducerBase();
 
    /// used by the fwk to register list of products
    boost::function<void(const BranchDescription&)> registrationCallback() const;

    void registerProducts(boost::shared_ptr<ProducerBase>,
			ProductRegistry *,
			ModuleDescription const&,
			bool throwIfNoProducts);

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
