#ifndef L1TMuonEndCap_AngleCalculation_h
#define L1TMuonEndCap_AngleCalculation_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"


class AngleCalculation {
public:
  void configure(
      int verbose, int endcap, int sector, int bx,
      int bxWindow,
      int thetaWindow, int thetaWindowRPC,
      bool bugME11Dupes
  );

  void process(
      zone_array<EMTFTrackCollection>& zone_tracks
  ) const;

  void calculate_angles(EMTFTrack& track) const;

  void calculate_bx(EMTFTrack& track) const;

  void erase_tracks(EMTFTrackCollection& tracks) const;

private:
  int verbose_, endcap_, sector_, bx_;

  int bxWindow_;
  int thetaWindow_, thetaWindowRPC_;
  bool bugME11Dupes_;
};

#endif
