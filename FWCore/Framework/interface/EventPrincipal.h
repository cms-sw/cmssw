#ifndef Framework_EventPrincipal_h
#define Framework_EventPrincipal_h

/*----------------------------------------------------------------------
  
EventPrincipal: This is the class responsible for management of
per event EDProducts. It is not seen by reconstruction code;
such code sees the Event class, which is a proxy for EventPrincipal.

The major internal component of the EventPrincipal
is the DataBlock.

$Id: EventPrincipal.h,v 1.39 2006/10/23 23:51:56 chrjones Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EventAux.h"
#include "FWCore/Framework/interface/DataBlock.h"

namespace edm {
  typedef DataBlock<EventAux> EventPrincipal;
}
#endif
