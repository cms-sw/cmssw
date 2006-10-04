/*----------------------------------------------------------------------
  $Id: Selector.cc,v 1.3 2006/03/06 01:18:53 chrjones Exp $
  ----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Selector.h"

namespace edm
{

  //------------------------------------------------------------------
  //
  // SelectorBase
  //  
  //------------------------------------------------------------------
  SelectorBase::~SelectorBase()
  { }

  bool
  SelectorBase::match(ProvenanceAccess const& p) const
  {
    return doMatch(p.provenance());
  }

  bool
  SelectorBase::match(Provenance const& p) const
  {
    return doMatch(p);
  }

  //------------------------------------------------------------------
  //
  // Selector
  //  
  //------------------------------------------------------------------


  Selector::Selector(Selector const& other) :
    sel_(other.sel_->clone())
  { }
  
  Selector&
  Selector::operator= (Selector const& other)
  {
    Selector temp(other);
    swap(temp);
    return *this;
  }

  void
  Selector::swap(Selector& other)
  {
    std::swap(sel_, other.sel_);
  }

  // We set sel_ = 0 to help diagnose memory overwrites.
  Selector::~Selector() { delete sel_; sel_ = 0; }

  Selector*
  Selector::clone() const
  {
    return new Selector(*this);
  }


  bool
  Selector::doMatch(Provenance const& prov) const
  {
    return sel_->match(prov);
  }

}
