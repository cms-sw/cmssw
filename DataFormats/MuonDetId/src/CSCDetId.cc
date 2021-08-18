#include "DataFormats/MuonDetId/interface/CSCDetId.h"

int CSCDetId::triggerSector() const {
  // UPDATED TO OCT 2005 - LGRAY Feb 2006

  int result;
  int ring = this->ring();
  int station = this->station();
  int chamber = this->chamber();

  if (station > 1 && ring > 1) {
    result = ((static_cast<unsigned>(chamber - 3) & 0x7f) / 6) + 1;  // ch 3-8->1, 9-14->2, ... 1,2 -> 6
  } else {
    result = (station != 1) ? ((static_cast<unsigned>(chamber - 2) & 0x1f) / 3) + 1 :  // ch 2-4-> 1, 5-7->2, ...
                 ((static_cast<unsigned>(chamber - 3) & 0x7f) / 6) + 1;
  }

  // max sector is 6, some calculations give a value greater than six but this is expected.
  return (result <= 6) ? result : 6;
}

int CSCDetId::triggerCscId() const {
  // UPDATED TO OCT 2005 - LGRAY Feb 2006

  int result;
  int ring = this->ring();
  int station = this->station();
  int chamber = this->chamber();

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

unsigned short CSCDetId::iChamberType(unsigned short istation, unsigned short iring) {
  int i = 2 * istation + iring;  // i=2S+R ok for S=2, 3, 4
  if (istation == 1) {
    --i;  // ring 1R -> i=1+R (2S+R-1=1+R for S=1)
    if (i > 4)
      i = 1;  // But ring 1A (R=4) -> i=1
  }
  return i;
}

bool CSCDetId::isME1a() const { return iChamberType(station(), ring()) == 1; }
bool CSCDetId::isME1b() const { return iChamberType(station(), ring()) == 2; }
bool CSCDetId::isME11() const { return isME1a() or isME1b(); }
bool CSCDetId::isME12() const { return iChamberType(station(), ring()) == 3; }
bool CSCDetId::isME13() const { return iChamberType(station(), ring()) == 4; }
bool CSCDetId::isME21() const { return iChamberType(station(), ring()) == 5; }
bool CSCDetId::isME22() const { return iChamberType(station(), ring()) == 6; }
bool CSCDetId::isME31() const { return iChamberType(station(), ring()) == 7; }
bool CSCDetId::isME32() const { return iChamberType(station(), ring()) == 8; }
bool CSCDetId::isME41() const { return iChamberType(station(), ring()) == 9; }
bool CSCDetId::isME42() const { return iChamberType(station(), ring()) == 10; }

std::string CSCDetId::chamberName(int endcap, int station, int ring, int chamber) {
  const std::string eSign = endcap == 1 ? "+" : "-";
  return "ME" + eSign + std::to_string(station) + "/" + std::to_string(ring) + "/" + std::to_string(chamber);
}

std::string CSCDetId::chamberName(int chamberType) {
  // ME1a, ME1b, ME12, ME13, ME21, ME22, ME31, ME32, ME41, ME42
  const unsigned stations[10] = {1, 1, 1, 1, 2, 2, 3, 3, 4, 4};
  const std::string rings[10] = {"A", "B", "2", "3", "1", "2", "1", "2", "1", "2"};
  return "ME" + std::to_string(stations[chamberType - 1]) + rings[chamberType - 1];
}

std::string CSCDetId::chamberName() const { return chamberName(endcap(), station(), ring(), chamber()); }

std::ostream& operator<<(std::ostream& os, const CSCDetId& id) {
  // Note that there is no endl to end the output

  os << " E:" << id.endcap() << " S:" << id.station() << " R:" << id.ring() << " C:" << id.chamber()
     << " L:" << id.layer();
  return os;
}
