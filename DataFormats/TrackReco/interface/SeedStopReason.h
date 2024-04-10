#ifndef DataFormats_TrackReco_SeedStopReason_h
#define DataFormats_TrackReco_SeedStopReason_h

#include <string>

enum class SeedStopReason {
  UNINITIALIZED = 0,
  NOT_STOPPED = 1,
  SEED_CLEANING = 2,
  NO_TRAJECTORY = 3,
  SEED_REGION_REBUILD = 4,
  FINAL_CLEAN = 5,
  SMOOTHING_FAILED = 6,
  SIZE = 7
};

namespace SeedStopReasonName {
  extern const std::string SeedStopReasonName[];
}

#endif
