#ifndef RecoTracker_MkFitCore_src_MkFitTbb_h

#define RecoTracker_MkFitCore_src_MkFitTbb_h

// Convert TBB execution to simple loops for debugging, perfomance measeurements.

#ifdef TBB_DEBUG

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/partitioner.h"

#define TBB_PARALLEL_FOR mkfit_tbb::parallel_for
#define TBB_PARALLEL_FOR_EACH mkfit_tbb::parallel_for_each

namespace mkfit_tbb {

  template <typename Range, typename Body>
  void parallel_for(const Range& range, const Body& body) {
    typename Range::const_iterator step = range.grainsize();
    for (auto i = range.begin(); i < range.end(); i += step) {
      step = std::min(step, range.end() - i);
      body(Range(i, i + step, 1));
    }
  }

  template <typename Range, typename Body>
  void parallel_for(const Range& range, const Body& body, const tbb::simple_partitioner& partitioner) {
    typename Range::const_iterator step = range.grainsize();
    for (auto i = range.begin(); i < range.end(); i += step) {
      step = std::min(step, range.end() - i);
      body(Range(i, i + step, 1));
    }
  }

  template <typename InputIterator, typename Function>
  void parallel_for_each(InputIterator first, InputIterator last, const Function& f) {
    for (auto& i = first; i != last; ++i) {
      f(*i);
    }
  }

}  // namespace mkfit_tbb

#else

#define TBB_PARALLEL_FOR tbb::parallel_for
#define TBB_PARALLEL_FOR_EACH tbb::parallel_for_each

#endif

#endif
