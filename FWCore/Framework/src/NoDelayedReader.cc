/*----------------------------------------------------------------------
$Id: NoDelayedReader.cc,v 1.1 2005/09/28 05:20:11 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/NoDelayedReader.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {
  NoDelayedReader::~NoDelayedReader() {}

  std::auto_ptr<EDProduct>
  NoDelayedReader::get(BranchKey const& k, EventPrincipal const* ep) const {
    throw cms::Exception("LogicError","NoDelayedReader")
      << "get() called for branchkey: " << k << " EventID: " << ep->id() << "\n";
  }
}
