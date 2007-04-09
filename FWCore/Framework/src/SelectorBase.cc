/*----------------------------------------------------------------------
  $Id: SelectorBase.cc,v 1.1 2006/10/23 23:49:01 chrjones Exp $
  ----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/SelectorBase.h"
#include "FWCore/Framework/interface/SelectorProvenance.h"

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
  SelectorBase::match(Provenance const& p) const
  {
    SelectorProvenance sp(p);
    return doMatch(sp);
  }

  bool
  SelectorBase::match(SelectorProvenance const& p) const
  {
    return doMatch(p);
  }
}
