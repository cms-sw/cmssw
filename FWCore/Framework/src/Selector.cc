/*----------------------------------------------------------------------
$Id: Selector.cc,v 1.2 2005/03/25 16:59:14 paterno Exp $
----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/interface/Selector.h"

namespace edm
{
  Selector::~Selector()
  { }

  bool
  Selector::match(const Provenance& p) const
  {
    return doMatch(p);
  }
}
