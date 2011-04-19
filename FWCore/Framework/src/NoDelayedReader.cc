/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/NoDelayedReader.h"

#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  NoDelayedReader::~NoDelayedReader() {}

  WrapperHolder
  NoDelayedReader::getProduct_(BranchKey const& k, WrapperInterfaceBase const*, EDProductGetter const* ep) const {
    EventPrincipal const* epr = dynamic_cast<EventPrincipal const*>(ep);
    if (epr) {
      throw edm::Exception(errors::LogicError,"NoDelayedReader")
        << "getProduct() called for branchkey: " << k << " EventID: " << epr->id() << "\n";
    }
    RunPrincipal const* rpr = dynamic_cast<RunPrincipal const*>(ep);
    if (rpr) {
      throw edm::Exception(errors::LogicError,"NoDelayedReader")
        << "getProduct() called for branchkey: " << k << " RunID: " << epr->id() << "\n";
    }
    LuminosityBlockPrincipal const* lpr = dynamic_cast<LuminosityBlockPrincipal const*>(ep);
    if (lpr) {
      throw edm::Exception(errors::LogicError,"NoDelayedReader")
        << "getProduct() called for branchkey: " << k << " LuminosityBlockNumber_t: " << lpr->id() << "\n";
    }
    throw edm::Exception(errors::LogicError,"NoDelayedReader")
      << "getProduct() called for branchkey: " << k << "\n";
  }
}
