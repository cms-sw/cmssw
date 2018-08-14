#ifndef L1TMuonEndCap_BestTrackSelection_h
#define L1TMuonEndCap_BestTrackSelection_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"


class BestTrackSelection {
public:
  void configure(
      int verbose, int endcap, int sector, int bx,
      int bxWindow,
      int maxRoadsPerZone, int maxTracks, bool useSecondEarliest,
      bool bugSameSectorPt0
  );

  void process(
      const std::deque<EMTFTrackCollection>& extended_best_track_cands,
      EMTFTrackCollection& best_tracks
  ) const;

  void cancel_one_bx(
      const std::deque<EMTFTrackCollection>& extended_best_track_cands,
      EMTFTrackCollection& best_tracks
  ) const;

  void cancel_multi_bx(
      const std::deque<EMTFTrackCollection>& extended_best_track_cands,
      EMTFTrackCollection& best_tracks
  ) const;

private:
  int verbose_, endcap_, sector_, bx_;

  int bxWindow_;
  int maxRoadsPerZone_, maxTracks_;
  bool useSecondEarliest_;
  bool bugSameSectorPt0_;
};

#endif
