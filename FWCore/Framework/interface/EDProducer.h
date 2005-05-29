#ifndef EDM_EDPRODUCER_INCLUDED
#define EDM_EDPRODUCER_INCLUDED

/*----------------------------------------------------------------------
  
EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

$Id: EDProducer.h,v 1.4 2005/04/21 04:21:38 jbk Exp $

----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"

namespace edm
{
  class EDProducer
  {
  public:
    typedef EDProducer module_type;

    virtual ~EDProducer();
    virtual void produce(Event& e, EventSetup const& c) = 0;
  };
}

#endif // EDM_EDPRODUCER_INCLUDED
