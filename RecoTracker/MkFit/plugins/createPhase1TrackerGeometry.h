#ifndef RecoTracker_MkFit_plugins_createPhase1TrackerGeometry_h
#define RecoTracker_MkFit_plugins_createPhase1TrackerGeometry_h

namespace mkfit {
  class TrackerInfo;
  class IterationsInfo;

  void createPhase1TrackerGeometry(TrackerInfo &ti, IterationsInfo &ii, bool verbose);
}  // namespace mkfit

#endif
