#ifndef Framework_EDProducer_h
#define Framework_EDProducer_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

$Id: EDProducer.h,v 1.10 2005/10/11 19:32:32 chrjones Exp $


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Framework/src/TypeID.h"
#include "boost/bind.hpp"
#include "boost/function.hpp"
#include <string>
#include <utility>
namespace edm {
  class BranchDescription;
  class Event;
  class EventSetup;
  class ParameterSet;
  class EDProducer : public ProductRegistryHelper {
  public:
    typedef EDProducer ModuleType;

    EDProducer () : ProductRegistryHelper() {}
    virtual ~EDProducer();
    virtual void produce(Event& e, EventSetup const& c) = 0;
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
 
    template<class TProducer, class TMethod>
    void callWhenNewProductsRegistered(TProducer* iProd, TMethod iMethod){
       callWhenNewProductsRegistered_ = boost::bind(iMethod,iProd,_1);
    }
          
    /// used by the fwk to register list of products
    boost::function<void(const BranchDescription&)> registrationCallback() const;

  private:
    boost::function<void(const BranchDescription&)> callWhenNewProductsRegistered_;
  };


}

#endif
