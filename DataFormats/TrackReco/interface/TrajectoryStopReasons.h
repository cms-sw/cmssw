#ifndef TRAJECTORYSTOPREASONS_H
#define TRAJECTORYSTOPREASONS_H

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
  NOT_STOPPED = 255 // this is the max allowed since it will be streamed as type uint8_t
};

#endif
