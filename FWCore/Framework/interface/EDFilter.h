#ifndef EDM_EDFILTER_INCLUDED
#define EDM_EDFILTER_INCLUDED

/*----------------------------------------------------------------------
  
EDFilter: The base class of all "modules" used to control the flow of
processing in a processing path.

$Id: EDFilter.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $

----------------------------------------------------------------------*/
#include "FWCore/CoreFramework/interface/Event.h"

#include "FWCore/CoreFramework/interface/EventSetup.h"

namespace edm
  {
  class EDFilter
    {
    public:
      typedef EDFilter ModuleType;

      virtual ~EDFilter();
      virtual bool filter(Event const& e, EventSetup const& c) = 0;
    };
}

#endif //
