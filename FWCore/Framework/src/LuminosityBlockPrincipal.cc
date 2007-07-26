#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

namespace edm {

  LuminosityBlockPrincipal::LuminosityBlockPrincipal(LuminosityBlockNumber_t const& lb,
	Timestamp const& time,
	boost::shared_ptr<ProductRegistry const> reg,
        boost::shared_ptr<RunPrincipal> rp,
        ProcessConfiguration const& pc,
	ProcessHistoryID const& hist,
	boost::shared_ptr<DelayedReader> rtrv) :
        Base(reg, pc, hist, rtrv), runPrincipal_(rp), aux_(rp->run(), lb, time) {}

  LuminosityBlockPrincipal::LuminosityBlockPrincipal(LuminosityBlockNumber_t const& lb,
	Timestamp const& time,
	boost::shared_ptr<ProductRegistry const> reg,
        RunNumber_t run,
        ProcessConfiguration const& pc,
	ProcessHistoryID const& hist,
	boost::shared_ptr<DelayedReader> rtrv) :
        Base(reg, pc, hist, rtrv), runPrincipal_(new RunPrincipal(run, time, reg, pc)), aux_(run, lb, time) {}
}

