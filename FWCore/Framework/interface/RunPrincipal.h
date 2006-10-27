#ifndef Framework_RunPrincipal_h
#define Framework_RunPrincipal_h

/*----------------------------------------------------------------------
  
RunPrincipal: This is the class responsible for management of
per run EDProducts. It is not seen by reconstruction code;
such code sees the Run class, which is a proxy for RunPrincipal.

The major internal component of the RunPrincipal
is the DataBlock.

$Id: RunPrincipal.h,v 1.38 2006/10/13 01:45:21 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/RunAux.h"
#include "FWCore/Framework/interface/DataBlock.h"

namespace edm {
  typedef DataBlock<RunAux> RunPrincipal;
}
#endif
