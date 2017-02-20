#ifndef DataFormats_TrackReco_SeedStopReason_h
#define DataFormats_TrackReco_SeedStopReason_h

enum SeedStopReason {
  UNINITIALIZED = 0,
  NOT_STOPPED = 1,
  SEED_CLEANING = 2,
  NO_TRAJECTORY = 3,
  FINAL_CLEAN = 4,
  SMOOTHING_FAILED = 5,
  SIZE = 6
};

#endif
