#ifndef L1TMuonEndCap_AngleCalculation_h
#define L1TMuonEndCap_AngleCalculation_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"


class AngleCalculation {
public:
  void configure(
      int verbose, int endcap, int sector, int bx,
      int bxWindow,
      int thetaWindow, int thetaWindowZone0,
      bool bugME11Dupes, bool bugAmbigThetaWin, bool twoStationSameBX
  );

  void process(
      emtf::zone_array<EMTFTrackCollection>& zone_tracks
  ) const;

  void calculate_angles(EMTFTrack& track, const int izone) const;

  void calculate_bx(EMTFTrack& track) const;

  void erase_tracks(EMTFTrackCollection& tracks) const;

private:
  int verbose_, endcap_, sector_, bx_;

  int bxWindow_;
  int thetaWindow_, thetaWindowZone0_;
  bool bugME11Dupes_, bugAmbigThetaWin_, twoStationSameBX_;
};

#endif
