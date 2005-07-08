#ifndef EDM_EDPRODUCER_INCLUDED
#define EDM_EDPRODUCER_INCLUDED

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

$Id: EDProducer.h,v 1.2 2005/06/23 05:23:10 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"

namespace edm
{
  class EDProducer
  {
  public:
    typedef EDProducer ModuleType;

    virtual ~EDProducer();
    virtual void produce(Event& e, EventSetup const& c) = 0;
    virtual void beginJob( EventSetup const& ) ;
    virtual void endJob() ;
    
  };
}

#endif // EDM_EDPRODUCER_INCLUDED
