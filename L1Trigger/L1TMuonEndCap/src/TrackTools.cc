#include "L1Trigger/L1TMuonEndCap/interface/TrackTools.h"

namespace emtf {

int calc_ring(int station, int csc_ID, int strip) {
  if (station > 1) {
    if      (csc_ID <  4) return 1;
    else if (csc_ID < 10) return 2;
    else return -999;
  }
  else if (station == 1) {
    if      (csc_ID < 4 && strip > 127) return 4;
    else if (csc_ID < 4 && strip >=  0) return 1;
    else if (csc_ID > 3 && csc_ID <  7) return 2;
    else if (csc_ID > 6 && csc_ID < 10) return 3;
    else return -999;
  }
  else return -999;
}

int calc_chamber(int station, int sector, int subsector, int ring, int csc_ID) {
  int chamber = -999;
  if (station == 1) {
    chamber = ((sector-1) * 6) + csc_ID + 2; // Chamber offset of 2: First chamber in sector 1 is chamber 3
    if (ring == 2)      chamber -= 3;
    if (ring == 3)      chamber -= 6;
    if (subsector == 2) chamber += 3;
    if (chamber > 36)   chamber -= 36;
  }
  else if (ring == 1) {
    chamber = ((sector-1) * 3) + csc_ID + 1; // Chamber offset of 1: First chamber in sector 1 is chamber 2
    if (chamber > 18)   chamber -= 18;
  }
  else if (ring == 2) {
    chamber = ((sector-1) * 6) + csc_ID - 3 + 2; // Chamber offset of 2: First chamber in sector 1 is chamber 3
    if (chamber > 36)   chamber -= 36;
  }
  return chamber;
}

// Calculates special chamber ID for track address sent to uGMT, using CSC_ID, subsector, neighbor, and station
int calc_uGMT_chamber(int csc_ID, int subsector, int neighbor, int station) {
  if (station == 1) {
    if      (csc_ID == 3 && neighbor == 1 && subsector == 2) return 1;
    else if (csc_ID == 6 && neighbor == 1 && subsector == 2) return 2;
    else if (csc_ID == 9 && neighbor == 1 && subsector == 2) return 3;
    else if (csc_ID == 3 && neighbor == 0 && subsector == 2) return 4;
    else if (csc_ID == 6 && neighbor == 0 && subsector == 2) return 5;
    else if (csc_ID == 9 && neighbor == 0 && subsector == 2) return 6;
    else return 0;
  }
  else {
    if      (csc_ID == 3 && neighbor == 1) return 1;
    else if (csc_ID == 9 && neighbor == 1) return 2;
    else if (csc_ID == 3 && neighbor == 0) return 3;
    else if (csc_ID == 9 && neighbor == 0) return 4;
    else return 0;
  }
}

}  // namespace emtf
