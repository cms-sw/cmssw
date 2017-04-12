#ifndef L1TMuonEndCap_PrimitiveMatching_hh
#define L1TMuonEndCap_PrimitiveMatching_hh

#include "L1Trigger/L1TMuonEndCap/interface/Common.hh"


class PrimitiveMatching {
public:
  typedef EMTFHitCollection::const_iterator hit_ptr_t;
  typedef std::pair<int, hit_ptr_t> hit_sort_pair_t;  // key=ph_diff, value=hit

  void configure(
      int verbose, int endcap, int sector, int bx,
      bool fixZonePhi, bool useNewZones,
      bool bugME11Dupes
  );

  void process(
      const std::deque<EMTFHitCollection>& extended_conv_hits,
      const zone_array<EMTFRoadCollection>& zone_roads,
      zone_array<EMTFTrackCollection>& zone_tracks
  ) const;

  void process_single_zone_station(
      int zone, int station,
      const EMTFRoadCollection& roads,
      const EMTFHitCollection& conv_hits,
      std::vector<hit_sort_pair_t>& phi_differences
  ) const;

  void insert_hits(
      hit_ptr_t conv_hit_ptr, const EMTFHitCollection& conv_hits,
      EMTFTrack& track
  ) const;

private:
  int verbose_, endcap_, sector_, bx_;

  bool fixZonePhi_, useNewZones_;
  bool bugME11Dupes_;
};

#endif
