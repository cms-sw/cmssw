/*----------------------------------------------------------------------
$Id: NoDelayedReader.cc,v 1.6 2007/03/04 06:10:25 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/NoDelayedReader.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  NoDelayedReader::~NoDelayedReader() {}

  std::auto_ptr<EDProduct>
  NoDelayedReader::getProduct(BranchKey const& k, EDProductGetter const* ep) const {
    EventPrincipal const* epr = dynamic_cast<EventPrincipal const*>(ep);
    if (epr) {
      throw cms::Exception("LogicError","NoDelayedReader")
        << "getProduct() called for branchkey: " << k << " EventID: " << epr->id() << "\n";
    }
    RunPrincipal const* rpr = dynamic_cast<RunPrincipal const*>(ep);
    if (rpr) {
      throw cms::Exception("LogicError","NoDelayedReader")
        << "getProduct() called for branchkey: " << k << " RunID: " << epr->id() << "\n";
    }
    LuminosityBlockPrincipal const* lpr = dynamic_cast<LuminosityBlockPrincipal const*>(ep);
    if (lpr) {
      throw cms::Exception("LogicError","NoDelayedReader")
        << "getProduct() called for branchkey: " << k << " LuminosityBlockNumber_t: " << lpr->id() << "\n";
    }
    throw cms::Exception("LogicError","NoDelayedReader")
      << "getProduct() called for branchkey: " << k << "\n";
  }

  std::auto_ptr<BranchEntryDescription>
  NoDelayedReader::getProvenance(BranchKey const& k, EDProductGetter const* ep) const {
    EventPrincipal const* epr = dynamic_cast<EventPrincipal const*>(ep);
    if (epr) {
      throw cms::Exception("LogicError","NoDelayedReader")
        << "getProvenance() called for branchkey: " << k << " EventID: " << epr->id() << "\n";
    }
    RunPrincipal const* rpr = dynamic_cast<RunPrincipal const*>(ep);
    if (rpr) {
      throw cms::Exception("LogicError","NoDelayedReader")
        << "getProvenance() called for branchkey: " << k << " RunID: " << epr->id() << "\n";
    }
    LuminosityBlockPrincipal const* lpr = dynamic_cast<LuminosityBlockPrincipal const*>(ep);
    if (lpr) {
      throw cms::Exception("LogicError","NoDelayedReader")
        << "getProvenance() called for branchkey: " << k << " LuminosityBlockNumber_t: " << lpr->id() << "\n";
    }
    throw cms::Exception("LogicError","NoDelayedReader")
      << "getProvenance() called for branchkey: " << k << "\n";
  }
}
