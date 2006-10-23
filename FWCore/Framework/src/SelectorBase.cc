/*----------------------------------------------------------------------
  $Id: Selector.cc,v 1.4 2006/10/04 14:53:20 paterno Exp $
  ----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/SelectorBase.h"

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
}
