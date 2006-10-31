#ifndef Framework_EventPrincipal_h
#define Framework_EventPrincipal_h

/*----------------------------------------------------------------------
  
EventPrincipal: This is a typedef for the class responsible for
management of per event EDProducts. It is not seen by reconstruction code;
such code sees the Event class, which is a proxy for EventPrincipal.

$Id: EventPrincipal.h,v 1.40 2006/10/27 20:56:42 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EventAux.h"
#include "FWCore/Framework/interface/DataBlock.h"

namespace edm {
  typedef DataBlock<EventAux> EventPrincipal;
}
#endif
