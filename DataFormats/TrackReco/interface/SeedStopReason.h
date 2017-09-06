#ifndef DataFormats_TrackReco_SeedStopReason_h
#define DataFormats_TrackReco_SeedStopReason_h

#include <string>

enum class SeedStopReason {
  UNINITIALIZED = 0,
  NOT_STOPPED = 1,
  SEED_CLEANING = 2,
  NO_TRAJECTORY = 3,
  FINAL_CLEAN = 4,
  SMOOTHING_FAILED = 5,
  SIZE = 6
};

namespace SeedStopReasonName {
  static const std::string SeedStopReasonName[] = {
    "UNINITIALIZED",    //  0
    "NOT_STOPPED",      //  1
    "SEED_CLEANING",    //  2
    "NO_TRAJECTORY",    //  3
    "FINAL_CLEAN",      //  4
    "SMOOTHING_FAILED"  //  5
  };
}

#endif
