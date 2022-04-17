#include "Matrix.h"
//#define DEBUG
#include <Debug.h>

#include "MkFitter.h"

#ifndef NO_ROOT
#include "TFile.h"
#include "TTree.h"
#include <mutex>
#endif

#include "oneapi/tbb/parallel_for.h"

#include <iostream>
#include <memory>

#if defined(USE_VTUNE_PAUSE)
#include "ittnotify.h"
#endif

//==============================================================================
// runFittingTestPlex
//==============================================================================

#include "Pool.h"
namespace {
  struct ExecutionContext {
    mkfit::Pool<mkfit::MkFitter> m_fitters;

    void populate(int n_thr) { m_fitters.populate(n_thr - m_fitters.size()); }
  };

  ExecutionContext g_exe_ctx;
  auto retfitr = [](mkfit::MkFitter* mkfp) { g_exe_ctx.m_fitters.ReturnToPool(mkfp); };
}  // namespace

namespace mkfit {

  double runFittingTestPlex(Event& ev, std::vector<Track>& rectracks) {
    g_exe_ctx.populate(Config::numThreadsFinder);
    std::vector<Track>& simtracks = ev.simTracks_;

    const int Nhits = Config::nLayers;
    // XXX What if there's a missing / double layer?
    // Eventually, should sort track vector by number of hits!
    // And pass the number in on each "setup" call.
    // Reserves should be made for maximum possible number (but this is just
    // measurments errors, params).

    int theEnd = simtracks.size();
    int count = (theEnd + NN - 1) / NN;

#ifdef USE_VTUNE_PAUSE
    __SSC_MARK(0x111);  // use this to resume Intel SDE at the same point
    __itt_resume();
#endif

    double time = dtime();

    tbb::parallel_for(tbb::blocked_range<int>(0, count, std::max(1, Config::numSeedsPerTask / NN)),
                      [&](const tbb::blocked_range<int>& i) {
                        std::unique_ptr<MkFitter, decltype(retfitr)> mkfp(g_exe_ctx.m_fitters.GetFromPool(), retfitr);
                        mkfp->setNhits(Nhits);
                        for (int it = i.begin(); it < i.end(); ++it) {
                          int itrack = it * NN;
                          int end = itrack + NN;
                          /*
         * MT, trying to slurp and fit at the same time ...
	  if (theEnd < end) {
	    end = theEnd;
	    mkfp->inputTracksAndHits(simtracks, ev.layerHits_, itrack, end);
	  } else {
	    mkfp->slurpInTracksAndHits(simtracks, ev.layerHits_, itrack, end); // only safe for a full matriplex
	  }
	  
	  if (Config::cf_fitting) mkfp->ConformalFitTracks(true, itrack, end);
	  mkfp->FitTracks(end - itrack, &ev, true);
        */

                          mkfp->inputTracksForFit(simtracks, itrack, end);

                          // XXXX MT - for this need 3 points in ... right
                          // XXXX if (Config::cf_fitting) mkfp->ConformalFitTracks(true, itrack, end);

                          mkfp->fitTracksWithInterSlurp(ev.layerHits_, end - itrack);

                          mkfp->outputFittedTracks(rectracks, itrack, end);
                        }
                      });

    time = dtime() - time;

#ifdef USE_VTUNE_PAUSE
    __itt_pause();
    __SSC_MARK(0x222);  // use this to pause Intel SDE at the same point
#endif

    if (Config::fit_val)
      ev.validate();

    return time;
  }

}  // end namespace mkfit
