#ifndef DataFormats_TrackReco_SeedStopReason_h
#define DataFormats_TrackReco_SeedStopReason_h

// Using unscoped enum because all uses are casts to integer, so
// implicit casting is convenient
struct SeedStopReason {
  enum {
    UNINITIALIZED = 0,
    NOT_STOPPED = 1,
    SEED_CLEANING = 2,
    NO_TRAJECTORY = 3,
    FINAL_CLEAN = 4,
    SMOOTHING_FAILED = 5,
    SIZE = 6
  };
};

#endif
