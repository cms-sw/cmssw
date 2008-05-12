/*----------------------------------------------------------------------
  $Id: SelectorBase.cc,v 1.2.6.1 2008/05/12 15:33:09 wmtan Exp $
  ----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/SelectorBase.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"

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
  SelectorBase::match(ConstBranchDescription const& p) const
  {
    return doMatch(p);
  }
}
