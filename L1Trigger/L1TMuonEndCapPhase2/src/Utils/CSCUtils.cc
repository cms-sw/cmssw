#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/CSCUtils.h"

namespace emtf::phase2::csc {

  // Chambers
  int next_10deg_chamber(int chamber) { return (chamber == 36) ? 1 : (chamber + 1); }

  int prev_10deg_chamber(int chamber) { return (chamber == 1) ? 36 : (chamber - 1); }

  int next_20deg_chamber(int chamber) { return (chamber == 18) ? 1 : (chamber + 1); }

  int prev_20deg_chamber(int chamber) { return (chamber == 1) ? 18 : (chamber - 1); }

  // Sectors
  bool is_in_sector(int sp_endcap, int sp_sector, int tp_endcap, int tp_sector) {
    return sp_endcap == tp_endcap && sp_sector == tp_sector;
  }

  bool is_in_neighbor_sector(
      int sp_endcap, int sp_sector, int tp_endcap, int tp_sector, int tp_subsector, int tp_station, int tp_id) {
    // Match endcap and neighbor sector
    int neighbor_sector = ((sp_sector == 1) ? 6 : sp_sector - 1);

    if ((sp_endcap != tp_endcap) || (neighbor_sector != tp_sector))
      return false;

    // Match CSCID in station 1
    if (tp_station == 1)
      return (tp_subsector == 2) && (tp_id == 3 || tp_id == 6 || tp_id == 9);

    // Match CSCID in other stations
    return tp_id == 3 || tp_id == 9;
  }

  // Use CSC trigger "CSC ID" definitions
  // Copied from DataFormats/MuonDetId/src/CSCDetId.cc
  int get_id(int station, int ring, int chamber) {
    int result = 0;

    if (station == 1) {
      result = (chamber) % 3 + 1;  // 1,2,3

      switch (ring) {
        case 1:
          break;
        case 2:
          result += 3;  // 4,5,6
          break;
        case 3:
          result += 6;  // 7,8,9
          break;
        case 4:
          break;
      }
    } else {
      if (ring == 1) {
        result = (chamber + 1) % 3 + 1;  // 1,2,3
      } else {
        result = (chamber + 3) % 6 + 4;  // 4,5,6,7,8,9
      }
    }

    return result;
  }

  // Use CSC trigger sector definitions
  // Copied from DataFormats/MuonDetId/src/CSCDetId.cc
  int get_trigger_sector(int station, int ring, int chamber) {
    int result = 0;

    if (station > 1 && ring > 1) {
      result = ((static_cast<unsigned>(chamber - 3) & 0x7f) / 6) + 1;  // ch 3-8->1, 9-14->2, ... 1,2 -> 6
    } else if (station == 1) {
      result = ((static_cast<unsigned>(chamber - 3) & 0x7f) / 6) + 1;  // ch 3-8->1, 9-14->2, ... 1,2 -> 6
    } else {
      result = ((static_cast<unsigned>(chamber - 2) & 0x1f) / 3) + 1;  // ch 2-4-> 1, 5-7->2, ...
    }

    return (result <= 6) ? result
                         : 6;  // max sector is 6, some calculations give a value greater than six but this is expected.
  }

  int get_trigger_subsector(int station, int chamber) {
    // station 2,3,4 --> subsector 0
    if (station != 1) {
      return 0;
    }

    // station 1 --> subsector 1 or 2
    if ((chamber % 6) > 2) {
      return 1;
    }

    return 2;
  }

  // Copied from RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.cc
  Facing get_face_direction(int station, int ring, int chamber) {
    bool is_not_overlapping = (station == 1 && ring == 3);

    // Not overlapping means it's facing backwards
    if (is_not_overlapping)
      return Facing::kRear;

    // odd chambers are bolted to the iron, which faces
    // forward in stations 1 and 2, backward in stations 3 and 4
    bool is_even = (chamber % 2 == 0);

    if (station < 3)
      return (is_even ? Facing::kRear : Facing::kFront);

    return (is_even ? Facing::kFront : Facing::kRear);
  }

  // Number of halfstrips and wiregroups
  // +----------------------------+------------+------------+
  // | Chamber type               | Num of     | Num of     |
  // |                            | halfstrips | wiregroups |
  // +----------------------------+------------+------------+
  // | ME1/1a                     | 96         | 48         |
  // | ME1/1b                     | 128        | 48         |
  // | ME1/2                      | 160        | 64         |
  // | ME1/3                      | 128        | 32         |
  // | ME2/1                      | 160        | 112        |
  // | ME3/1, ME4/1               | 160        | 96         |
  // | ME2/2, ME3/2, ME4/2        | 160        | 64         |
  // +----------------------------+------------+------------+

  std::pair<int, int> get_max_strip_and_wire(int station, int ring) {
    int max_strip = 0;  // halfstrip
    int max_wire = 0;   // wiregroup

    if (station == 1 && ring == 4) {  // ME1/1a
      max_strip = 96;
      max_wire = 48;
    } else if (station == 1 && ring == 1) {  // ME1/1b
      max_strip = 128;
      max_wire = 48;
    } else if (station == 1 && ring == 2) {  // ME1/2
      max_strip = 160;
      max_wire = 64;
    } else if (station == 1 && ring == 3) {  // ME1/3
      max_strip = 128;
      max_wire = 32;
    } else if (station == 2 && ring == 1) {  // ME2/1
      max_strip = 160;
      max_wire = 112;
    } else if (station >= 3 && ring == 1) {  // ME3/1, ME4/1
      max_strip = 160;
      max_wire = 96;
    } else if (station >= 2 && ring == 2) {  // ME2/2, ME3/2, ME4/2
      max_strip = 160;
      max_wire = 64;
    }

    return std::make_pair(max_strip, max_wire);
  }

  std::pair<int, int> get_max_pattern_and_quality(int station, int ring) {
    int max_pattern = 11;
    int max_quality = 16;

    return std::make_pair(max_pattern, max_quality);
  }

}  // namespace emtf::phase2::csc
