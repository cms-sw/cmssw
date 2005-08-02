#ifndef Framework_EDProducer_h
#define Framework_EDProducer_h

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

$Id: EDProducer.h,v 1.6 2005/07/22 23:53:18 wmtan Exp $


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/TypeID.h"
#include <string>
#include <utility>
namespace edm {
  class EDProducer {
  public:
    typedef EDProducer ModuleType;

    virtual ~EDProducer();
    virtual void produce(Event& e, EventSetup const& c) = 0;
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
 
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
    TypeLabelList typeLabelList() const;

  private:
    TypeLabelList typeLabelList_;
  };


}

#endif // EDM_EDPRODUCER_INCLUDED
