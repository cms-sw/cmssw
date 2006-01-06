/*----------------------------------------------------------------------
$Id: NoDelayedReader.cc,v 1.2 2005/12/01 22:14:54 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/NoDelayedReader.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {
  NoDelayedReader::~NoDelayedReader() {}

  std::auto_ptr<EDProduct>
  NoDelayedReader::get(BranchKey const& k, EDProductGetter const* ep) const {
    EventPrincipal const* epr = dynamic_cast<EventPrincipal const*>(ep);
    throw cms::Exception("LogicError","NoDelayedReader")
      << "get() called for branchkey: " << k << " EventID: " << (epr ? epr->id() : EventID()) << "\n";
  }
}
