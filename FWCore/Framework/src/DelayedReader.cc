#include "FWCore/Framework/interface/DelayedReader.h"

#include <cassert>
/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/


namespace edm {
  DelayedReader::~DelayedReader() {}

  WrapperOwningHolder
  DelayedReader::getProductInStream_(BranchKey const&, WrapperInterfaceBase const*, EDProductGetter const*, StreamID const&) const {
    assert(0 && "This delayed reader does not support streams");
    return WrapperOwningHolder();
  }
}
