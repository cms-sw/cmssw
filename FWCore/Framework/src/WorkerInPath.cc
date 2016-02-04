

#include "FWCore/Framework/src/WorkerInPath.h"

namespace edm {
  WorkerInPath::WorkerInPath(Worker* w, FilterAction theFilterAction):
    stopwatch_(),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    filterAction_(theFilterAction),
    worker_(w)
  {
  }

  WorkerInPath::WorkerInPath(Worker* w):
    stopwatch_(new RunStopwatch::StopwatchPointer::element_type),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    filterAction_(Normal),
    worker_(w)
  {
  }
   
   void 
   WorkerInPath::useStopwatch() {
      stopwatch_.reset(new RunStopwatch::StopwatchPointer::element_type);
   }


}
