#ifndef Framework_ProducerBase_h
#define Framework_ProducerBase_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

$Id: ProducerBase.h,v 1.12 2006/02/20 01:51:57 wmtan Exp $


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "boost/bind.hpp"
#include "boost/function.hpp"
#include <string>
#include <utility>
namespace edm {
  class BranchDescription;
  class ModuleDescription;
  class ProductRegistry;
  class ProducerBase : public ProductRegistryHelper {
  public:
    ProducerBase ();
    virtual ~ProducerBase();
 
    /// used by the fwk to register list of products
    boost::function<void(const BranchDescription&)> registrationCallback() const;

    void registerProducts(boost::shared_ptr<ProducerBase>,
			ProductRegistry *,
			ModuleDescription const&,
			bool throwIfNoProducts);

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
