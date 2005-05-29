#ifndef EDM_EDFILTER_INCLUDED
#define EDM_EDFILTER_INCLUDED

/*----------------------------------------------------------------------
  
EDFilter: The base class of all "modules" used to control the flow of
processing in a processing path.

$Id: EDFilter.h,v 1.4 2005/04/21 04:21:38 jbk Exp $

----------------------------------------------------------------------*/
#include "FWCore/CoreFramework/interface/Event.h"

#include "FWCore/CoreFramework/interface/EventSetup.h"

namespace edm
  {
  class EDFilter
    {
    public:
      typedef EDFilter module_type;

      virtual ~EDFilter();
      virtual bool filter(Event const& e, EventSetup const& c) = 0;
    };
}

#endif //
