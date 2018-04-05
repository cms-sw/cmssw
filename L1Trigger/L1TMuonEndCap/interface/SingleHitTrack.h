#ifndef L1TMuonEndCap_SingleHitTrack_h
#define L1TMuonEndCap_SingleHitTrack_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"


class SingleHitTrack {
public:
  void configure(
      int verbose, int endcap, int sector, int bx,
      int maxTracks,
      bool useSingleHits
  );

  void process(
      const EMTFHitCollection& conv_hits,
      EMTFTrackCollection& best_tracks
  ) const;


private:
  int verbose_, endcap_, sector_, bx_;
  int maxTracks_;
  bool useSingleHits_;
};

#endif
