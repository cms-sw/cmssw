#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

#include "boost/shared_ptr.hpp"

namespace edm {
  LuminosityBlockPrincipal::LuminosityBlockPrincipal(LuminosityBlockID const& id,
	ProductRegistry const& reg,
        ProcessConfiguration const& pc,
	ProcessHistoryID const& hist,
	boost::shared_ptr<DelayedReader> rtrv) :
        Base(reg, pc, hist, rtrv), runPrincipal_(new RunPrincipal(1, reg, pc)), aux_(id) {}
}
