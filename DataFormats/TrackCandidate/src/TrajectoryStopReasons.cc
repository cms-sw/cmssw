#include "DataFormats/TrackCandidate/interface/TrajectoryStopReasons.h"

const std::string StopReasonName::StopReasonName[] = {
    "UNINITIALIZED",                 //  0
    "MAX_HITS",                      //  1
    "MAX_LOST_HITS",                 //  2
    "MAX_CONSECUTIVE_LOST_HITS",     //  3
    "LOST_HIT_FRACTION",             //  4
    "MIN_PT",                        //  5
    "CHARGE_SIGNIFICANCE",           //  6
    "LOOPER",                        //  7
    "MAX_CCC_LOST_HITS",             //  8
    "NO_SEGMENTS_FOR_VALID_LAYERS",  //  9
    "SEED_EXTENSION",                // 10
    "NOT_STOPPED"                    // 11 (be careful, NOT_STOPPED needs to be the last,
                                     //     its index differs from the enumeration value)
};

static_assert(sizeof(StopReasonName::StopReasonName) / sizeof(std::string) ==
                  static_cast<unsigned int>(StopReason::SIZE),
              "StopReason enum and StopReasonName are out of synch");
