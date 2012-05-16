/*----------------------------------------------------------------------
  $Id: Selector.cc,v 1.8 2009/04/15 23:22:30 wmtan Exp $
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

  bool
  Selector::doMatchSelectorType(std::type_info const& type) const {
    return sel_->matchSelectorType(type);
  }

}
