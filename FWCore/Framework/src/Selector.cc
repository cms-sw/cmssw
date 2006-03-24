/*----------------------------------------------------------------------
$Id: Selector.cc,v 1.2 2005/07/14 22:50:53 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Selector.h"

namespace edm
{
  Selector::~Selector()
  { }

  bool
  Selector::match(const ProvenanceAccess& p) const
  {
    return doMatch(p);
  }
}
