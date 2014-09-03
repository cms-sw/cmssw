

#include "FWCore/Framework/src/WorkerInPath.h"

namespace edm {
  WorkerInPath::WorkerInPath(Worker* w, FilterAction theFilterAction, unsigned int placeInPath):
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    filterAction_(theFilterAction),
    worker_(w),
    placeInPathContext_(placeInPath)
  {
  }

}
