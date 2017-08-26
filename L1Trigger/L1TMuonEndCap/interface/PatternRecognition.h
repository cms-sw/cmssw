#ifndef L1TMuonEndCap_PatternRecognition_h
#define L1TMuonEndCap_PatternRecognition_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"
#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"


class PatternRecognition {
public:
  // Pattern detector ID: [zone, keystrip, pattern]
  typedef std::array<int, 3>  pattern_ref_t;

  void configure(
      int verbose, int endcap, int sector, int bx,
      int bxWindow,
      const std::vector<std::string>& pattDefinitions, const std::vector<std::string>& symPattDefinitions, bool useSymPatterns,
      int maxRoadsPerZone, bool useSecondEarliest
  );

  void configure_details();

  void process(
      const std::deque<EMTFHitCollection>& extended_conv_hits,
      std::map<pattern_ref_t, int>& patt_lifetime_map,
      zone_array<EMTFRoadCollection>& zone_roads
  ) const;

  bool is_zone_empty(
      int zone,
      const std::deque<EMTFHitCollection>& extended_conv_hits,
      const std::map<pattern_ref_t, int>& patt_lifetime_map
  ) const;

  void make_zone_image(
      int zone,
      const std::deque<EMTFHitCollection>& extended_conv_hits,
      PhiMemoryImage& image
  ) const;

  void process_single_zone(
      int zone,
      PhiMemoryImage cloned_image,
      std::map<pattern_ref_t, int>& patt_lifetime_map,
      EMTFRoadCollection& roads
  ) const;

  void sort_single_zone(EMTFRoadCollection& roads) const;


private:
  int verbose_, endcap_, sector_, bx_;

  int bxWindow_;
  std::vector<std::string> pattDefinitions_, symPattDefinitions_;
  bool useSymPatterns_;
  int maxRoadsPerZone_;
  bool useSecondEarliest_;

  std::vector<PhiMemoryImage> patterns_;
};

#endif
