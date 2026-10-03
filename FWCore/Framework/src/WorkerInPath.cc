

#include "FWCore/Framework/interface/WorkerInPath.h"

namespace edm {
  WorkerInPath::WorkerInPath(GlobalEventWorker* w,
                             FilterAction theFilterAction,
                             unsigned int placeInPath,
                             bool runConcurrently)
      : filterAction_(theFilterAction),
        worker_(w),
        placeInPathContext_(placeInPath),
        runConcurrently_(runConcurrently) {
    w->addedToPath();
  }

}  // namespace edm
