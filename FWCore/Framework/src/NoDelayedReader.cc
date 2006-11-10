/*----------------------------------------------------------------------
$Id: NoDelayedReader.cc,v 1.3 2006/01/06 00:29:32 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/NoDelayedReader.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

namespace edm {
  NoDelayedReader::~NoDelayedReader() {}

  std::auto_ptr<EDProduct>
  NoDelayedReader::get(BranchKey const& k, EDProductGetter const* ep) const {
    EventPrincipal const* epr = dynamic_cast<EventPrincipal const*>(ep);
    if (epr) {
      throw cms::Exception("LogicError","NoDelayedReader")
        << "get() called for branchkey: " << k << " EventID: " << epr->id() << "\n";
    }
    RunPrincipal const* rpr = dynamic_cast<RunPrincipal const*>(ep);
    if (rpr) {
      throw cms::Exception("LogicError","NoDelayedReader")
        << "get() called for branchkey: " << k << " RunID: " << epr->id() << "\n";
    }
    LuminosityBlockPrincipal const* lpr = dynamic_cast<LuminosityBlockPrincipal const*>(ep);
    if (lpr) {
      throw cms::Exception("LogicError","NoDelayedReader")
        << "get() called for branchkey: " << k << " LuminosityBlockID: " << lpr->id() << "\n";
    }
    throw cms::Exception("LogicError","NoDelayedReader")
      << "get() called for branchkey: " << k << "\n";
  }
}
