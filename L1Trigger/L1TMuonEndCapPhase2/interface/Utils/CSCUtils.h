#ifndef L1Trigger_L1TMuonEndCapPhase2_CSCUtils_h
#define L1Trigger_L1TMuonEndCapPhase2_CSCUtils_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"

namespace emtf::phase2::csc {

  // Enums
  enum Facing { kFront, kRear, kNone };

  // Chambers
  int getNext10DegChamber(int chamber);

  int getPrev10DegChamber(int chamber);

  int getNext20DegChamber(int chamber);

  int getPrev20DegChamber(int chamber);

  // Functions
  bool isTPInSector(int match_endcap, int match_sector, int tp_endcap, int tp_sector);

  bool isTPInNeighborSector(
      int match_endcap, int match_sector, int tp_endcap, int tp_sector, int tp_subsector, int tp_station, int tp_id);

  int getId(int ring, int station, int chamber);

  int getTriggerSector(int ring, int station, int chamber);

  int getTriggerSubsector(int station, int chamber);

  Facing getFaceDirection(int station, int ring, int chamber);

  std::pair<int, int> getMaxStripAndWire(int station, int ring);

  std::pair<int, int> getMaxPatternAndQuality(int station, int ring);

}  // namespace emtf::phase2::csc

#endif  // namespace L1Trigger_L1TMuonEndCapPhase2_CSCUtils_h
