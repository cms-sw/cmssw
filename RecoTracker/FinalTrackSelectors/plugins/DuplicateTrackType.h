#ifndef RecoTracker_FinalTrackSelectors_DuplicateTrackType_h
#define RecoTracker_FinalTrackSelectors_DuplicateTrackType_h

enum class DuplicateTrackType {
  NotDuplicate = 0,  // default
  Disjoint,          // tracks are disjoint
  Overlapping,       // tracks overlap
};

#endif
