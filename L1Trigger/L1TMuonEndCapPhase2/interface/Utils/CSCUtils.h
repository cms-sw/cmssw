#ifndef L1Trigger_L1TMuonEndCapPhase2_CSCUtils_h
#define L1Trigger_L1TMuonEndCapPhase2_CSCUtils_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"

namespace emtf::phase2::csc {

  // Enums
  enum Facing { kFront, kRear, kNone };

  // Chambers
  int next_10deg_chamber(int chamber);

  int prev_10deg_chamber(int chamber);

  int next_20deg_chamber(int chamber);

  int prev_20deg_chamber(int chamber);

  // Functions
  bool is_in_sector(int match_endcap, int match_sector, int tp_endcap, int tp_sector);

  bool is_in_neighbor_sector(
      int match_endcap, int match_sector, int tp_endcap, int tp_sector, int tp_subsector, int tp_station, int tp_id);

  int get_id(int ring, int station, int chamber);

  int get_trigger_sector(int ring, int station, int chamber);

  int get_trigger_subsector(int station, int chamber);

  Facing get_face_direction(int station, int ring, int chamber);

  std::pair<int, int> get_max_strip_and_wire(int station, int ring);

  std::pair<int, int> get_max_pattern_and_quality(int station, int ring);

}  // namespace emtf::phase2::csc

#endif  // namespace L1Trigger_L1TMuonEndCapPhase2_CSCUtils_h
