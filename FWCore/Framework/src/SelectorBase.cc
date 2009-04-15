
#include "FWCore/Framework/interface/SelectorBase.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"

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
