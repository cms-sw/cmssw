
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

  bool
  SelectorBase::matchSelectorType(std::type_info const& type) const {
    return doMatchSelectorType(type);
  }
}
