#ifndef Framework_LuminosityBlockPrincipal_h
#define Framework_LuminosityBlockPrincipal_h

/*----------------------------------------------------------------------
  
LuminosityBlockPrincipal: This is the class responsible for management of
per LuminosityBlock EDProducts. It is not seen by reconstruction code;
such code sees the LuminosityBlock class, which is a proxy for LuminosityBlockPrincipal.

The major internal component of the LuminosityBlockPrincipal
is the DataBlock.

$Id: LuminosityBlockPrincipal.h,v 1.38 2006/10/13 01:45:21 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/LuminosityBlockAux.h"
#include "FWCore/Framework/interface/DataBlock.h"

namespace edm {
  typedef DataBlock<LuminosityBlockAux> LuminosityBlockPrincipal;
}
#endif
