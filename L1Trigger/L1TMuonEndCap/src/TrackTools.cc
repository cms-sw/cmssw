#include "L1Trigger/L1TMuonEndCap/interface/TrackTools.h"

namespace emtf {

  int calc_ring(int station, int csc_ID, int strip) {
    if (station > 1) {
      if (csc_ID < 4)
        return 1;
      else if (csc_ID < 10)
        return 2;
      else
        return -999;
    } else if (station == 1) {
      if (csc_ID < 4 && strip > 127)
        return 4;
      else if (csc_ID < 4 && strip >= 0)
        return 1;
      else if (csc_ID > 3 && csc_ID < 7)
        return 2;
      else if (csc_ID > 6 && csc_ID < 10)
        return 3;
      else
        return -999;
    } else
      return -999;
  }

  int calc_chamber(int station, int sector, int subsector, int ring, int csc_ID) {
    int chamber = -999;
    if (station == 1) {
      chamber = ((sector - 1) * 6) + csc_ID + 2;  // Chamber offset of 2: First chamber in sector 1 is chamber 3
      if (ring == 2)
        chamber -= 3;
      if (ring == 3)
        chamber -= 6;
      if (subsector == 2)
        chamber += 3;
      if (chamber > 36)
        chamber -= 36;
    } else if (ring == 1) {
      chamber = ((sector - 1) * 3) + csc_ID + 1;  // Chamber offset of 1: First chamber in sector 1 is chamber 2
      if (chamber > 18)
        chamber -= 18;
    } else if (ring == 2) {
      chamber = ((sector - 1) * 6) + csc_ID - 3 + 2;  // Chamber offset of 2: First chamber in sector 1 is chamber 3
      if (chamber > 36)
        chamber -= 36;
    }
    return chamber;
  }

  // Calculates special chamber ID for track address sent to uGMT, using CSC_ID, subsector, neighbor, and station
  int calc_uGMT_chamber(int csc_ID, int subsector, int neighbor, int station) {
    if (station == 1) {
      if (csc_ID == 3 && neighbor == 1 && subsector == 2)
        return 1;
      else if (csc_ID == 6 && neighbor == 1 && subsector == 2)
        return 2;
      else if (csc_ID == 9 && neighbor == 1 && subsector == 2)
        return 3;
      else if (csc_ID == 3 && neighbor == 0 && subsector == 2)
        return 4;
      else if (csc_ID == 6 && neighbor == 0 && subsector == 2)
        return 5;
      else if (csc_ID == 9 && neighbor == 0 && subsector == 2)
        return 6;
      else
        return 0;
    } else {
      if (csc_ID == 3 && neighbor == 1)
        return 1;
      else if (csc_ID == 9 && neighbor == 1)
        return 2;
      else if (csc_ID == 3 && neighbor == 0)
        return 3;
      else if (csc_ID == 9 && neighbor == 0)
        return 4;
      else
        return 0;
    }
  }

  // Use CSC trigger sector definitions
  // Copied from DataFormats/MuonDetId/src/CSCDetId.cc
  int get_trigger_sector(int ring, int station, int chamber) {
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

  // Use CSC trigger "CSC ID" definitions
  // Copied from DataFormats/MuonDetId/src/CSCDetId.cc
  int get_trigger_csc_ID(int ring, int station, int chamber) {
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
        case 4:                            // ME0
          result = (chamber + 1) % 3 + 1;  // 1,2,3
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

  std::pair<int, int> get_csc_max_strip_and_wire(int station, int ring) {
    int max_strip = 0;                // halfstrip
    int max_wire = 0;                 // wiregroup
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

  std::pair<int, int> get_csc_max_pattern_and_quality(int station, int ring) {
    int max_pattern = 11;
    int max_quality = 16;
    return std::make_pair(max_pattern, max_quality);
  }

  int get_csc_max_slope(int station, int ring, bool useRun3CCLUT_OTMB, bool useRun3CCLUT_TMB) {
    int max_slope = 65536;  // Uninitialized slope can be 65536. This is expected when CCLUT is not running
    if (useRun3CCLUT_OTMB and (ring == 1 or ring == 4))
      max_slope = 16;
    if (useRun3CCLUT_TMB and (ring == 2 or ring == 3))
      max_slope = 16;
    return max_slope;
  }

}  // namespace emtf
