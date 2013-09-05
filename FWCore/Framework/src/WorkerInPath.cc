

#include "FWCore/Framework/src/WorkerInPath.h"

namespace edm {
  WorkerInPath::WorkerInPath(Worker* w, FilterAction theFilterAction, unsigned int placeInPath):
    stopwatch_(),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    filterAction_(theFilterAction),
    worker_(w),
    placeInPathContext_(placeInPath)
  {
  }

   void 
   WorkerInPath::useStopwatch() {
      stopwatch_.reset(new RunStopwatch::StopwatchPointer::element_type);
   }


}
