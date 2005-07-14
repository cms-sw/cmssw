#ifndef EDM_EDFILTER_INCLUDED
#define EDM_EDFILTER_INCLUDED

/*----------------------------------------------------------------------
  
EDFilter: The base class of all "modules" used to control the flow of
processing in a processing path.

$Id: EDFilter.h,v 1.3 2005/07/08 00:09:38 chrjones Exp $

----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"

namespace edm
  {
  class EDFilter
    {
    public:
      typedef EDFilter ModuleType;

      virtual ~EDFilter();
      virtual bool filter(Event const& e, EventSetup const& c) = 0;
      virtual void beginJob( EventSetup const& ) ;
      virtual void endJob() ;
      
      
    };
}

#endif //
