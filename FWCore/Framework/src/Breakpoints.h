#ifndef FWCore_Framework_Breakpoints_h
#define FWCore_Framework_Breakpoints_h
/*
  Functions used only as breakpoints only to aid debugging.

  They cannot be optimized away because they are
  in a separate compilation unit.

  Performance is not an issue because each of these functions
  is called only once per job.

*/

namespace bk {
  void
  beginJob();

  void
  beginRuns();
}

#endif
