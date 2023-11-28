#include "DataFormats/TrackReco/interface/SeedStopReason.h"

const std::string SeedStopReasonName::SeedStopReasonName[] = {
    "UNINITIALIZED",        // 0
    "NOT_STOPPED",          // 1
    "SEED_CLEANING",        // 2
    "NO_TRAJECTORY",        // 3
    "SEED_REGION_REBUILD",  // 4
    "FINAL_CLEAN",          // 5
    "SMOOTHING_FAILED"      // 6
};

static_assert(sizeof(SeedStopReasonName::SeedStopReasonName) / sizeof(std::string) ==
                  static_cast<unsigned int>(SeedStopReason::SIZE),
              "SeedStopReason enum and SeedStopReasonName are out of synch");
