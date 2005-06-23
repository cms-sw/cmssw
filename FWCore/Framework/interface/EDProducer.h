#ifndef EDM_EDPRODUCER_INCLUDED
#define EDM_EDPRODUCER_INCLUDED

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

$Id: EDProducer.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $

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
  };
}

#endif // EDM_EDPRODUCER_INCLUDED
