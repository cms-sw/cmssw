#ifndef EDM_EDPRODUCER_INCLUDED
#define EDM_EDPRODUCER_INCLUDED

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

$Id: EDProducer.h,v 1.4 2005/07/14 22:50:52 wmtan Exp $


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/TypeID.h"
#include <string>
#include <utility>
namespace edm
{
  class EDProducer
  {
  public:
    typedef EDProducer ModuleType;

    virtual ~EDProducer();
    virtual void produce(Event& e, EventSetup const& c)=0;
    virtual void beginJob( EventSetup const& ) ;
    virtual void endJob() ;
 
    /// declare what type of product will make and with which optional label 
    /** the statement
        \code
           produces<ProductType>("optlabel");
        \endcode
        should be added to the producer ctor for every product */

    template <class ProductType> 
    void  produces( std::string label="" ){

      ProductType aproduct;
      productList_.push_back(std::make_pair(TypeID(aproduct), label));
      
    }

    typedef std::list<std::pair<TypeID,std::string> > TypeLabelList;
    /// used by the fwk to register the list of products of this module 
    const TypeLabelList& getTypeLabelList() const;

  private:
    TypeLabelList productList_;
  };


}

#endif // EDM_EDPRODUCER_INCLUDED
