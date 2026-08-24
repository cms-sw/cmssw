

#include "FWCore/Framework/interface/WorkerInPath.h"

namespace edm {
  WorkerInPath::WorkerInPath(Worker* w, FilterAction theFilterAction, unsigned int placeInPath, bool runConcurrently)
      : filterAction_(theFilterAction),
        worker_(w),
        placeInPathContext_(placeInPath),
        runConcurrently_(runConcurrently) {
    w->addedToPath();
  }

}  // namespace edm
