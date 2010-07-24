
#include "FWCore/Framework/interface/SelectorBase.h"

namespace edm {

  //------------------------------------------------------------------
  //
  // SelectorBase
  //  
  //------------------------------------------------------------------
  SelectorBase::~SelectorBase()
  { }

  bool
  SelectorBase::match(ConstBranchDescription const& p) const {
    return doMatch(p);
  }
}
