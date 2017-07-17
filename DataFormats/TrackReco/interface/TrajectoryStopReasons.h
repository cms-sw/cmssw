#ifndef TRAJECTORYSTOPREASONS_H
#define TRAJECTORYSTOPREASONS_H

#include <string>

enum class StopReason {
  UNINITIALIZED = 0,
  MAX_HITS = 1,
  MAX_LOST_HITS = 2,
  MAX_CONSECUTIVE_LOST_HITS = 3,
  LOST_HIT_FRACTION = 4,
  MIN_PT = 5,
  CHARGE_SIGNIFICANCE = 6,
  LOOPER = 7,
  MAX_CCC_LOST_HITS = 8,
  NO_SEGMENTS_FOR_VALID_LAYERS = 9,
  SEED_EXTENSION = 10,
  SIZE = 12, // This gives the number of the stopping reasons. The cound needs to be manually maintained, and should be 2 + the last value above .
  NOT_STOPPED = 255 // this is the max allowed since it will be streamed as type uint8_t
};


// to be kept in synch w/ the above enum ;)
namespace StopReasonName {
  static const std::string StopReasonName[] = {
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
};

#endif
