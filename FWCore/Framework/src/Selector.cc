/*----------------------------------------------------------------------
  $Id: Selector.cc,v 1.7 2008/05/12 18:14:08 wmtan Exp $
  ----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Selector.h"

namespace edm
{
  //------------------------------------------------------------------
  //
  // Selector
  //  
  //------------------------------------------------------------------

  void
  Selector::swap(Selector& other) {
    std::swap(sel_, other.sel_);
  }

  Selector::~Selector() { }

  Selector*
  Selector::clone() const {
    return new Selector(*this);
  }

  bool
  Selector::doMatch(ConstBranchDescription const& prov) const {
    return sel_->match(prov);
  }

}
