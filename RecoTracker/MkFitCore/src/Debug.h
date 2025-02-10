#ifndef RecoTracker_MkFitCore_src_Debug_h

namespace mkfit {
  extern bool g_debug;
}

#ifdef DEBUG
#define RecoTracker_MkFitCore_src_Debug_h

#ifdef dprint

#undef dprint
#undef dprint_np
#undef dcall
#undef dprintf
#undef dprintf_np

#endif
/*
  Usage: DEBUG must be defined before this header file is included, typically

  #define DEBUG
  #include "Debug.h"

  This defines macros dprint(), dcall() and dprintf();
  dprint(x) is equivalent to std::cout << x << std::endl;
    example: dprint("Hits in layer=" << ilayer);

  dcall(x) simply calls x
    example: dcall(pre_prop_print(ilay, mkfp));

  dprintf(x) is equivalent to printf(x)
    example: dprintf("Bad label for simtrack %d -- %d\n", itrack, track.label());

  All printouts are also controlled by a bool variable "debug"
  bool debug = true; is declared as a file global in an anonymous
  namespace, and thus can be overridden within any interior scope
  as needed, so one could change the global to false and only set
  a local to true within certain scopes.

  All are protected by a file scope mutex to avoid mixed printouts.
  This mutex can also be acquired within a block via dmutex_guard:

  if (debug && g_debug) {
    dmutex_guard;
    [do complicated stuff]
  }

  The mutex is not reentrant, so avoid using dprint et al. within a scope
  where the mutex has already been acquired, as doing so will deadlock.
 */
#include <mutex>

#define dmutex_guard std::lock_guard<std::mutex> dlock(debug_mutex)
#define dprint(x)                \
  if (debug && g_debug) {        \
    dmutex_guard;                \
    std::cout << x << std::endl; \
  }
#define dprint_np(n, x)                       \
  if (debug && g_debug && n < N_proc) {       \
    dmutex_guard;                             \
    std::cout << n << ": " << x << std::endl; \
  }
#define dcall(x)          \
  if (debug && g_debug) { \
    dmutex_guard;         \
    x;                    \
  }
#define dprintf(...)      \
  if (debug && g_debug) { \
    dmutex_guard;         \
    printf(__VA_ARGS__);  \
  }
#define dprintf_np(n, ...)              \
  if (debug && g_debug && n < N_proc) { \
    dmutex_guard;                       \
    std::cout << n << ": ";             \
    printf(__VA_ARGS__);                \
  }

namespace {
  bool debug = false;  // default, can be overridden locally
  std::mutex debug_mutex;

  struct debug_guard {
    bool m_prev_debug;
    debug_guard(bool state = true) : m_prev_debug(debug) { debug = state; }
    ~debug_guard() { debug = m_prev_debug; }
  };
}  // namespace

#else

#define dprint(x) (void(0))
#define dprint_np(n, x) (void(0))
#define dcall(x) (void(0))
#define dprintf(...) (void(0))
#define dprintf_np(n, ...) (void(0))

#endif

// Convert TBB execution to simple loops for debugging, perfomance measeurements.

#ifdef TBB_DISABLE
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
