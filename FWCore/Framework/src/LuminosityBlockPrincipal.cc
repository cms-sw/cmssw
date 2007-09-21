#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

namespace edm {

  LuminosityBlockPrincipal::LuminosityBlockPrincipal(LuminosityBlockNumber_t const& lb,
	Timestamp const& beginTm,
	Timestamp const& endTm,
	boost::shared_ptr<ProductRegistry const> reg,
        boost::shared_ptr<RunPrincipal> rp,
        ProcessConfiguration const& pc,
	ProcessHistoryID const& hist,
	boost::shared_ptr<DelayedReader> rtrv) :
          Base(reg, pc, hist, rtrv),
	  runPrincipal_(rp), aux_(rp->run(), lb, beginTm, endTm) {}
}
