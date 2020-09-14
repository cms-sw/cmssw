#include "L1Trigger/L1TMuonEndCap/interface/PatternRecognition.h"

#include "helper.h"  // to_hex, to_binary

namespace {
  const int padding_w_st1 = 15;
  const int padding_w_st3 = 7;
  const int padding_extra_w_st1 = padding_w_st1 - padding_w_st3;
}  // namespace

void PatternRecognition::configure(int verbose,
                                   int endcap,
                                   int sector,
                                   int bx,
                                   int bxWindow,
                                   const std::vector<std::string>& pattDefinitions,
                                   const std::vector<std::string>& symPattDefinitions,
                                   bool useSymPatterns,
                                   int maxRoadsPerZone,
                                   bool useSecondEarliest) {
  verbose_ = verbose;
  endcap_ = endcap;
  sector_ = sector;
  bx_ = bx;

  bxWindow_ = bxWindow;
  pattDefinitions_ = pattDefinitions;
  symPattDefinitions_ = symPattDefinitions;
  useSymPatterns_ = useSymPatterns;
  maxRoadsPerZone_ = maxRoadsPerZone;
  useSecondEarliest_ = useSecondEarliest;

  configure_details();
}

void PatternRecognition::configure_details() {
  emtf_assert(1 <= bxWindow_ && bxWindow_ <= 3);  // only work for BX windows <= 3

  patterns_.clear();

  // Parse pattern definitions
  if (!useSymPatterns_) {
    // Normal patterns
    for (const auto& s : pattDefinitions_) {
      const std::vector<std::string>& tokens = split_string(s, ',', ':');  // split by comma or colon
      emtf_assert(tokens.size() == 9);                                     // want to find 9 numbers

      std::vector<std::string>::const_iterator tokens_it = tokens.begin();

      // Get the 9 integers
      // straightness, hits in ME1, hits in ME2, hits in ME3, hits in ME4
      int straightness = std::stoi(*tokens_it++);
      int st1_max = std::stoi(*tokens_it++);
      int st1_min = std::stoi(*tokens_it++);
      int st2_max = std::stoi(*tokens_it++);
      int st2_min = std::stoi(*tokens_it++);
      int st3_max = std::stoi(*tokens_it++);
      int st3_min = std::stoi(*tokens_it++);
      int st4_max = std::stoi(*tokens_it++);
      int st4_min = std::stoi(*tokens_it++);

      // There can only be one zone hit in the key station in the pattern
      // and it has to be this magic number
      emtf_assert(st2_max == padding_w_st3 && st2_min == padding_w_st3);

      // There is extra "padding" in st1 w.r.t st2,3,4
      // Add the extra padding to st2,3,4
      st2_max += padding_extra_w_st1;
      st2_min += padding_extra_w_st1;
      st3_max += padding_extra_w_st1;
      st3_min += padding_extra_w_st1;
      st4_max += padding_extra_w_st1;
      st4_min += padding_extra_w_st1;

      // Create a pattern
      PhiMemoryImage pattern;
      pattern.set_straightness(straightness);
      int i = 0;

      for (i = st1_min; i <= st1_max; i++)
        pattern.set_bit(0, i);
      for (i = st2_min; i <= st2_max; i++)
        pattern.set_bit(1, i);
      for (i = st3_min; i <= st3_max; i++)
        pattern.set_bit(2, i);
      for (i = st4_min; i <= st4_max; i++)
        pattern.set_bit(3, i);

      // Remove the extra padding
      pattern.rotr(padding_extra_w_st1);
      patterns_.push_back(pattern);
    }
    emtf_assert(patterns_.size() == pattDefinitions_.size());

  } else {
    // Symmetrical patterns
    for (const auto& s : symPattDefinitions_) {
      const std::vector<std::string>& tokens = split_string(s, ',', ':');  // split by comma or colon
      emtf_assert(tokens.size() == 17);                                    // want to find 17 numbers

      std::vector<std::string>::const_iterator tokens_it = tokens.begin();

      // Get the 17 integers
      // straightness, hits in ME1, hits in ME2, hits in ME3, hits in ME4
      int straightness = std::stoi(*tokens_it++);
      int st1_max1 = std::stoi(*tokens_it++);
      int st1_min1 = std::stoi(*tokens_it++);
      int st1_max2 = std::stoi(*tokens_it++);
      int st1_min2 = std::stoi(*tokens_it++);
      int st2_max1 = std::stoi(*tokens_it++);
      int st2_min1 = std::stoi(*tokens_it++);
      int st2_max2 = std::stoi(*tokens_it++);
      int st2_min2 = std::stoi(*tokens_it++);
      int st3_max1 = std::stoi(*tokens_it++);
      int st3_min1 = std::stoi(*tokens_it++);
      int st3_max2 = std::stoi(*tokens_it++);
      int st3_min2 = std::stoi(*tokens_it++);
      int st4_max1 = std::stoi(*tokens_it++);
      int st4_min1 = std::stoi(*tokens_it++);
      int st4_max2 = std::stoi(*tokens_it++);
      int st4_min2 = std::stoi(*tokens_it++);

      // There can only be one zone hit in the key station in the pattern
      // and it has to be this magic number
      emtf_assert(st2_max1 == padding_w_st3 && st2_min1 == padding_w_st3);
      emtf_assert(st2_max2 == padding_w_st3 && st2_min2 == padding_w_st3);

      // There is extra "padding" in st1 w.r.t st2,3,4
      // Add the extra padding to st2,3,4
      st2_max1 += padding_extra_w_st1;
      st2_min1 += padding_extra_w_st1;
      st2_max2 += padding_extra_w_st1;
      st2_min2 += padding_extra_w_st1;
      st3_max1 += padding_extra_w_st1;
      st3_min1 += padding_extra_w_st1;
      st3_max2 += padding_extra_w_st1;
      st3_min2 += padding_extra_w_st1;
      st4_max1 += padding_extra_w_st1;
      st4_min1 += padding_extra_w_st1;
      st4_max2 += padding_extra_w_st1;
      st4_min2 += padding_extra_w_st1;

      // Create a pattern
      PhiMemoryImage pattern;
      pattern.set_straightness(straightness);
      int i = 0;

      for (i = st1_min1; i <= st1_max1; i++)
        pattern.set_bit(0, i);
      for (i = st1_min2; i <= st1_max2; i++)
        pattern.set_bit(0, i);
      for (i = st2_min1; i <= st2_max1; i++)
        pattern.set_bit(1, i);
      for (i = st2_min2; i <= st2_max2; i++)
        pattern.set_bit(1, i);
      for (i = st3_min1; i <= st3_max1; i++)
        pattern.set_bit(2, i);
      for (i = st3_min2; i <= st3_max2; i++)
        pattern.set_bit(2, i);
      for (i = st4_min1; i <= st4_max1; i++)
        pattern.set_bit(3, i);
      for (i = st4_min2; i <= st4_max2; i++)
        pattern.set_bit(3, i);

      // Remove the extra padding
      pattern.rotr(padding_extra_w_st1);
      patterns_.push_back(pattern);
    }
    emtf_assert(patterns_.size() == symPattDefinitions_.size());
  }

  if (verbose_ > 2) {  // debug
    for (const auto& pattern : patterns_) {
      std::cout << "Pattern straightness: " << pattern.get_straightness() << " image: " << std::endl;
      std::cout << pattern << std::endl;
    }
  }
}

void PatternRecognition::process(const std::deque<EMTFHitCollection>& extended_conv_hits,
                                 std::map<pattern_ref_t, int>& patt_lifetime_map,
                                 emtf::zone_array<EMTFRoadCollection>& zone_roads) const {
  // Exit if no hits
  int num_conv_hits = 0;
  for (const auto& conv_hits : extended_conv_hits)
    num_conv_hits += conv_hits.size();
  bool early_exit = (num_conv_hits == 0) && (patt_lifetime_map.empty());

  if (early_exit)
    return;

  if (verbose_ > 0) {  // debug
    for (const auto& conv_hits : extended_conv_hits) {
      for (const auto& conv_hit : conv_hits) {
        if (conv_hit.Subsystem() == TriggerPrimitive::kCSC) {
          std::cout << "CSC hit st: " << conv_hit.PC_station() << " ch: " << conv_hit.PC_chamber()
                    << " ph: " << conv_hit.Phi_fp() << " th: " << conv_hit.Theta_fp() << " strip: " << conv_hit.Strip()
                    << " wire: " << conv_hit.Wire() << " cpat: " << conv_hit.Pattern()
                    << " zone_hit: " << conv_hit.Zone_hit() << " zone_code: " << conv_hit.Zone_code()
                    << " bx: " << conv_hit.BX() << std::endl;
        }
      }
    }

    for (const auto& conv_hits : extended_conv_hits) {
      for (const auto& conv_hit : conv_hits) {
        if (conv_hit.Subsystem() == TriggerPrimitive::kRPC) {
          std::cout << "RPC hit st: " << conv_hit.PC_station() << " ch: " << conv_hit.PC_chamber()
                    << " ph>>2: " << (conv_hit.Phi_fp() >> 2) << " th>>2: " << (conv_hit.Theta_fp() >> 2)
                    << " strip: " << conv_hit.Strip() << " roll: " << conv_hit.Roll() << " cpat: " << conv_hit.Pattern()
                    << " bx: " << conv_hit.BX() << std::endl;
        }
      }
    }

    for (const auto& conv_hits : extended_conv_hits) {
      for (const auto& conv_hit : conv_hits) {
        if (conv_hit.Subsystem() == TriggerPrimitive::kGEM) {
          std::cout << "GEM hit st: " << conv_hit.PC_station() << " ch: " << conv_hit.PC_chamber()
                    << " ph: " << conv_hit.Phi_fp() << " th: " << conv_hit.Theta_fp() << " strip: " << conv_hit.Strip()
                    << " roll: " << conv_hit.Roll() << " cpat: " << conv_hit.Pattern() << " bx: " << conv_hit.BX()
                    << std::endl;
        }
      }
    }
  }  // end debug

  // Perform pattern recognition in each zone
  emtf::zone_array<PhiMemoryImage> zone_images;

  for (int izone = 0; izone < emtf::NUM_ZONES; ++izone) {
    // Skip the zone if no hits and no patterns
    if (is_zone_empty(izone + 1, extended_conv_hits, patt_lifetime_map))
      continue;

    // Make zone images
    make_zone_image(izone + 1, extended_conv_hits, zone_images.at(izone));

    // Detect patterns
    process_single_zone(izone + 1, zone_images.at(izone), patt_lifetime_map, zone_roads.at(izone));
  }

  if (verbose_ > 2) {  // debug
    for (int izone = emtf::NUM_ZONES; izone >= 1; --izone) {
      std::cout << "zone: " << izone << std::endl;
      std::cout << zone_images.at(izone - 1) << std::endl;
    }
    //for (const auto& kv : patt_lifetime_map) {
    //  std::cout << "zone: " << kv.first.at(0) << " izhit: " << kv.first.at(1) << " ipatt: " << kv.first.at(2) << " lifetime: " << kv.second << std::endl;
    //}
  }

  if (verbose_ > 0) {  // debug
    for (const auto& roads : zone_roads) {
      for (const auto& road : reversed(roads)) {
        std::cout << "pattern: z: " << road.Zone() - 1 << " ph: " << road.Key_zhit()
                  << " q: " << to_hex(road.Quality_code()) << " ly: " << to_binary(road.Layer_code(), 3)
                  << " str: " << to_binary(road.Straightness(), 3) << " bx: " << road.BX() << std::endl;
      }
    }
  }

  // Sort patterns and select best three patterns in each zone
  for (int izone = 0; izone < emtf::NUM_ZONES; ++izone) {
    sort_single_zone(zone_roads.at(izone));
  }
}

bool PatternRecognition::is_zone_empty(int zone,
                                       const std::deque<EMTFHitCollection>& extended_conv_hits,
                                       const std::map<pattern_ref_t, int>& patt_lifetime_map) const {
  int izone = zone - 1;
  int num_conv_hits = 0;
  int num_patts = 0;

  std::deque<EMTFHitCollection>::const_iterator ext_conv_hits_it = extended_conv_hits.begin();
  std::deque<EMTFHitCollection>::const_iterator ext_conv_hits_end = extended_conv_hits.end();

  for (; ext_conv_hits_it != ext_conv_hits_end; ++ext_conv_hits_it) {
    EMTFHitCollection::const_iterator conv_hits_it = ext_conv_hits_it->begin();
    EMTFHitCollection::const_iterator conv_hits_end = ext_conv_hits_it->end();

    for (; conv_hits_it != conv_hits_end; ++conv_hits_it) {
      emtf_assert(conv_hits_it->PC_segment() <=
                  4);  // With 2 unique LCTs per chamber, 4 possible strip/wire combinations

      if (conv_hits_it->Subsystem() == TriggerPrimitive::kRPC)
        continue;  // Don't use RPC hits for pattern formation

      if (conv_hits_it->Subsystem() == TriggerPrimitive::kGEM)
        continue;  // Don't use GEM hits for pattern formation

      if (conv_hits_it->Zone_code() & (1 << izone)) {  // hit belongs to this zone
        num_conv_hits += 1;
      }
    }  // end loop over conv_hits
  }    // end loop over extended_conv_hits

  std::map<pattern_ref_t, int>::const_iterator patt_lifetime_map_it = patt_lifetime_map.begin();
  std::map<pattern_ref_t, int>::const_iterator patt_lifetime_map_end = patt_lifetime_map.end();

  for (; patt_lifetime_map_it != patt_lifetime_map_end; ++patt_lifetime_map_it) {
    if (patt_lifetime_map_it->first.at(0) == zone) {
      num_patts += 1;
    }
  }  // end loop over patt_lifetime_map

  return (num_conv_hits == 0) && (num_patts == 0);
}

void PatternRecognition::make_zone_image(int zone,
                                         const std::deque<EMTFHitCollection>& extended_conv_hits,
                                         PhiMemoryImage& image) const {
  int izone = zone - 1;

  std::deque<EMTFHitCollection>::const_iterator ext_conv_hits_it = extended_conv_hits.begin();
  std::deque<EMTFHitCollection>::const_iterator ext_conv_hits_end = extended_conv_hits.end();

  for (; ext_conv_hits_it != ext_conv_hits_end; ++ext_conv_hits_it) {
    EMTFHitCollection::const_iterator conv_hits_it = ext_conv_hits_it->begin();
    EMTFHitCollection::const_iterator conv_hits_end = ext_conv_hits_it->end();

    for (; conv_hits_it != conv_hits_end; ++conv_hits_it) {
      if (conv_hits_it->Subsystem() == TriggerPrimitive::kRPC)
        continue;  // Don't use RPC hits for pattern formation

      if (conv_hits_it->Subsystem() == TriggerPrimitive::kGEM)
        continue;  // Don't use GEM hits for pattern formation

      if (conv_hits_it->Zone_code() & (1 << izone)) {  // hit belongs to this zone
        unsigned int layer = conv_hits_it->Station() - 1;
        unsigned int bit = conv_hits_it->Zone_hit();
        image.set_bit(layer, bit);
      }
    }  // end loop over conv_hits
  }    // end loop over extended_conv_hits
}

void PatternRecognition::process_single_zone(int zone,
                                             PhiMemoryImage cloned_image,
                                             std::map<pattern_ref_t, int>& patt_lifetime_map,
                                             EMTFRoadCollection& roads) const {
  roads.clear();

  const int drift_time = bxWindow_ - 1;
  const int npatterns = patterns_.size();

  // The zone hit image is rotated/shifted before comparing with patterns
  // First, rotate left/shift up to the zone hit in the key station
  cloned_image.rotl(padding_w_st3);

  for (int izhit = 0; izhit < emtf::NUM_ZONE_HITS; ++izhit) {
    // The zone hit image is rotated/shift before comparing with patterns
    // For every zone hit, rotate right/shift down by one
    if (izhit > 0)
      cloned_image.rotr(1);

    int max_quality_code = -1;
    EMTFRoad tmp_road;

    // Compare with patterns
    for (int ipatt = 0; ipatt < npatterns; ++ipatt) {
      const PhiMemoryImage& patt = patterns_.at(ipatt);
      const pattern_ref_t patt_ref = {{zone, izhit, ipatt}};  // due to GCC bug, use {{}} instead of {}
      int straightness = patt.get_straightness();

      bool is_lifetime_up = false;

      int layer_code = patt.op_and(cloned_image);  // kind of like AND operator
      bool more_than_one = (layer_code != 0) && (layer_code != 1) && (layer_code != 2) && (layer_code != 4);
      bool more_than_zero = (layer_code != 0);

      if (more_than_zero) {
        // Insert this pattern
        auto ins = patt_lifetime_map.insert({patt_ref, 0});

        if (!useSecondEarliest_) {
          // Use earliest
          auto patt_ins = ins.first;  // iterator of patt_lifetime_map pointing to this pattern
          bool patt_exists = !ins.second;

          if (patt_exists) {                       // if exists
            if (patt_ins->second == drift_time) {  // is lifetime up?
              is_lifetime_up = true;
            }
          }
          patt_ins->second += 1;  // bx starts counting at any hit in the pattern, even single

        } else {
          // Use 2nd earliest
          auto patt_ins = ins.first;  // iterator of patt_lifetime_map pointing to this pattern
          int bx_shifter = patt_ins->second;
          int bx2 = bool(bx_shifter & (1 << 2));
          int bx1 = bool(bx_shifter & (1 << 1));
          int bx0 = bool(bx_shifter & (1 << 0));

          // is lifetime up?
          if (drift_time == 2 && bx2 == 0 && bx1 == 1) {
            is_lifetime_up = true;
          } else if (drift_time == 1 && bx1 == 0 && bx0 == 1) {
            is_lifetime_up = true;
          } else if (drift_time == 0) {
            is_lifetime_up = true;
          }

          bx2 = bx1;
          bx1 = bx0;
          bx0 = more_than_zero;  // put 1 in shifter when one layer is hit
          bx_shifter = (bx2 << 2) | (bx1 << 1) | bx0;
          patt_ins->second = bx_shifter;
        }

      } else {
        // Zero hit
        patt_lifetime_map.erase(patt_ref);  // erase if exists
      }

      // If lifetime is up, and not single-layer hit patterns (stations 3&4 considered
      // as a single layer), find quality of this pattern
      if (is_lifetime_up && more_than_one) {
        // This quality code scheme is giving almost-equal priority to
        // more stations and better straightness
        // Station 1 has higher weight, station 2 lower, stations 3&4 lowest
        int quality_code =
            ((((straightness >> 2) & 1) << 5) | (((straightness >> 1) & 1) << 3) | (((straightness >> 0) & 1) << 1) |
             (((layer_code >> 2) & 1) << 4) | (((layer_code >> 1) & 1) << 2) | (((layer_code >> 0) & 1) << 0));

        // Create a road (fired pattern)
        EMTFRoad road;
        road.set_endcap((endcap_ == 1) ? 1 : -1);
        road.set_sector(sector_);
        road.set_sector_idx((endcap_ == 1) ? sector_ - 1 : sector_ + 5);
        road.set_bx(bx_ - drift_time);

        road.set_zone(patt_ref.at(0));
        road.set_key_zhit(patt_ref.at(1));
        road.set_pattern(patt_ref.at(2));

        road.set_straightness(straightness);
        road.set_layer_code(layer_code);
        road.set_quality_code(quality_code);

        // Find max quality code in a given key_zhit
        if (max_quality_code < quality_code) {
          max_quality_code = quality_code;
          tmp_road = road;
        }
      }  // end if is_lifetime_up

    }  // end loop over patterns

    // Output road
    if (max_quality_code != -1) {
      roads.push_back(tmp_road);
    }

  }  // end loop over zone hits

  // Ghost cancellation logic by considering neighbor patterns

  if (!roads.empty()) {
    std::array<int, emtf::NUM_ZONE_HITS> quality_codes;
    quality_codes.fill(0);

    EMTFRoadCollection::iterator roads_it = roads.begin();
    EMTFRoadCollection::iterator roads_end = roads.end();

    for (; roads_it != roads_end; ++roads_it) {
      quality_codes.at(roads_it->Key_zhit()) = roads_it->Quality_code();
    }

    roads_it = roads.begin();
    roads_end = roads.end();

    for (; roads_it != roads_end; ++roads_it) {
      int izhit = roads_it->Key_zhit();

      // Center quality is the current one
      int qc = quality_codes.at(izhit);

      // Left and right qualities are the neighbors
      // Protect against the right end and left end special cases
      int ql = (izhit == emtf::NUM_ZONE_HITS - 1) ? 0 : quality_codes.at(izhit + 1);
      int qr = (izhit == 0) ? 0 : quality_codes.at(izhit - 1);

      // Cancellation conditions
      if (qc <= ql || qc < qr) {        // this pattern is lower quality than neighbors
        roads_it->set_quality_code(0);  // cancel
      }
    }
  }

  // Erase roads with quality_code == 0
  // using erase-remove idiom
  struct {
    typedef EMTFRoad value_type;
    bool operator()(const value_type& x) const { return (x.Quality_code() == 0); }
  } quality_code_zero_pred;

  roads.erase(std::remove_if(roads.begin(), roads.end(), quality_code_zero_pred), roads.end());
}

void PatternRecognition::sort_single_zone(EMTFRoadCollection& roads) const {
  // First, order by key_zhit (highest to lowest)
  struct {
    typedef EMTFRoad value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const { return lhs.Key_zhit() > rhs.Key_zhit(); }
  } greater_zhit_cmp;

  std::sort(roads.begin(), roads.end(), greater_zhit_cmp);

  // Second, sort by quality_code (highest to lowest), but preserving the original order if qualities are equal
  struct {
    typedef EMTFRoad value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      return lhs.Quality_code() > rhs.Quality_code();
    }
  } greater_quality_cmp;

  std::stable_sort(roads.begin(), roads.end(), greater_quality_cmp);

  // Finally, select 3 best
  const size_t n = maxRoadsPerZone_;
  if (roads.size() > n) {
    roads.erase(roads.begin() + n, roads.end());
  }
  emtf_assert(roads.size() <= n);

  // Assign the winner variable
  for (unsigned iroad = 0; iroad < roads.size(); ++iroad) {
    roads.at(iroad).set_winner(iroad);
  }
}
