#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <FWCore/Utilities/interface/Exception.h>

int CSCTriggerNumbering::ringFromTriggerLabels(int station, int triggerCSCID) {
  if (station < CSCDetId::minStationId() || station > CSCDetId::maxStationId() || triggerCSCID < MIN_CSCID ||
      triggerCSCID > MAX_CSCID)
    throw cms::Exception("CSCTriggerNumbering::InvalidInput")
        << "CSCTriggerNumbering::ringFromTriggerLabels():"
        << " Station: " << station << " TriggerCSCID: " << triggerCSCID << " is not a valid set of labels."
        << " Cannot Convert!!";

  int ring = 0;

  if (station == 1)
    if (triggerCSCID <= 3)
      ring = 1;
    else if (triggerCSCID <= 6)
      ring = 2;
    else
      ring = 3;
  else if (triggerCSCID <= 3)
    ring = 1;
  else
    ring = 2;

  return ring;
}

int CSCTriggerNumbering::chamberFromTriggerLabels(int TriggerSector,
                                                  int TriggerSubSector,
                                                  int station,
                                                  int TriggerCSCID) {
  if (TriggerSector < MIN_TRIGSECTOR || TriggerSector > MAX_TRIGSECTOR || TriggerSubSector < MIN_TRIGSUBSECTOR ||
      TriggerSubSector > MAX_TRIGSUBSECTOR || station < CSCDetId::minStationId() ||
      station > CSCDetId::maxStationId() || TriggerCSCID < MIN_CSCID || TriggerCSCID > MAX_CSCID)
    throw cms::Exception("CSCTriggerNumbering::InvalidInput")
        << "CSCTriggerNumbering::chamberFromTriggerLabels():"
        << " Trigger Sector: " << TriggerSector << " Trigger SubSector: " << TriggerSubSector << " Station: " << station
        << " TriggerCSCID: " << TriggerCSCID << " is not a valid set of labels."
        << " Cannot Convert!!";

  int chamber = 0;
  int realsubsector = (TriggerSubSector + 2 * (TriggerSector - 1)) % 12 + 1;  // station 1 only

  if (station != 1)
    if (TriggerCSCID <= 3)
      // Derived from CMS Note: CMS IN 2000/04 ver 2.1 Oct/2005
      // As far as I know this is reality.
      chamber = (TriggerCSCID + 3 * (TriggerSector - 1)) % 18 + 1;

    else
      chamber = (TriggerCSCID + 6 * (TriggerSector - 1) - 2) % 36 + 1;
  else if (TriggerCSCID <= 3)
    chamber = (TriggerCSCID + 3 * (realsubsector - 1) + 34) % 36 + 1;
  else if (TriggerCSCID <= 6)
    chamber = (TriggerCSCID + 3 * (realsubsector - 1) + 31) % 36 + 1;
  else
    chamber = (TriggerCSCID + 3 * (realsubsector - 1) + 28) % 36 + 1;

  return chamber;
}

int CSCTriggerNumbering::sectorFromTriggerLabels(int TriggerSector, int TriggerSubSector, int station) {
  if (TriggerSector < MIN_TRIGSECTOR || TriggerSector > MAX_TRIGSECTOR || TriggerSubSector < MIN_TRIGSUBSECTOR ||
      TriggerSubSector > MAX_TRIGSUBSECTOR || station < CSCDetId::minStationId() || station > CSCDetId::maxStationId())
    throw cms::Exception("CSCTriggerNumbering::InvalidInput")
        << "CSCTriggerNumbering::sectorFromTriggerLabels():"
        << " Trigger Sector: " << TriggerSector << " Trigger SubSector: " << TriggerSubSector << " Station: " << station
        << " is not a valid set of labels."
        << " Cannot Convert!!";

  return ((station == 1) ? ((TriggerSubSector + 2 * (TriggerSector - 1)) % 12 + 1) : TriggerSector);
}

int CSCTriggerNumbering::triggerSectorFromLabels(int station, int ring, int chamber) {
  if (station < CSCDetId::minStationId() || station > CSCDetId::maxStationId() || ring < CSCDetId::minRingId() ||
      ring > CSCDetId::maxRingId() || chamber < CSCDetId::minChamberId() || chamber > CSCDetId::maxChamberId())
    throw cms::Exception("CSCTriggerNumbering::InvalidInput")
        << "CSCTriggerNumbering::triggerSectorFromLabels():"
        << " Station: " << station << " Ring: " << ring << " Chamber: " << chamber << " is not a valid set of labels."
        << " Cannot Convert!!";

  int result;
  // This version 16-Nov-99 ptc to match simplified chamber labelling for cms116
  //@@ REQUIRES UPDATE TO 2005 REALITY, ONCE I UNDERSTAND WHAT THAT IS
  // UPDATED - LGRAY Feb 2006

  if (station > 1 && ring > 1) {
    result = ((static_cast<unsigned>(chamber - 3) & 0x7f) / 6) + 1;  // ch 3-8->1, 9-14->2, ... 1,2 -> 6
  } else {
    result = (station != 1) ? ((static_cast<unsigned>(chamber - 2) & 0x1f) / 3) + 1 :  // ch 2-4-> 1, 5-7->2, ...
                 ((static_cast<unsigned>(chamber - 3) & 0x7f) / 6) + 1;
  }

  // Max sector is 6, some calculations give a value greater than six but this is expected
  // and delt with.
  return (result <= 6) ? result : 6;
}

int CSCTriggerNumbering::triggerSectorFromLabels(CSCDetId id) {
  return triggerSectorFromLabels(id.station(), id.ring(), id.chamber());
}

int CSCTriggerNumbering::triggerSubSectorFromLabels(int station, int chamber) {
  if (station < CSCDetId::minStationId() || station > CSCDetId::maxStationId() || chamber < CSCDetId::minChamberId() ||
      chamber > CSCDetId::maxChamberId())
    throw cms::Exception("CSCTriggerNumbering::InvalidInput")
        << "CSCTriggerNumbering::triggerSectorFromLabels():"
        << " Station: " << station << " Chamber: " << chamber << " is not a valid set of labels."
        << " Cannot Convert!!";

  if (station != 1)
    return 0;  // only station one has subsectors

  switch (chamber)  // first make things easier to deal with
  {
    case 1:
      chamber = 36;
      break;
    case 2:
      chamber = 35;
      break;
    default:
      chamber -= 2;
  }

  chamber = ((chamber - 1) % 6) + 1;  // renumber all chambers to 1-6

  return ((chamber - 1) / 3) + 1;  // [1,3] -> 1 , [4,6]->2
}

int CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId id) {
  return triggerSubSectorFromLabels(id.station(), id.chamber());
}

int CSCTriggerNumbering::triggerCscIdFromLabels(int station, int ring, int chamber)  // updated to 2005
{
  if (station < CSCDetId::minStationId() || station > CSCDetId::maxStationId() || ring < CSCDetId::minRingId() ||
      ring > CSCDetId::maxRingId() || chamber < CSCDetId::minChamberId() || chamber > CSCDetId::maxChamberId())
    throw cms::Exception("CSCTriggerNumbering::InvalidInput")
        << "CSCTriggerNumbering::triggerSectorFromLabels():"
        << " Station: " << station << " Ring: " << ring << " Chamber: " << chamber << " is not a valid set of labels."
        << " Cannot Convert!!";

  int result;

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

int CSCTriggerNumbering::triggerCscIdFromLabels(CSCDetId id) {
  return triggerCscIdFromLabels(id.station(), id.ring(), id.chamber());
}
