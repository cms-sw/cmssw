#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <iostream>

int testCell() {
  int nerr(0);
  for (int re = GEMDetId::minRegionId; re <= GEMDetId::maxRegionId; ++re) {
    if (re != 0) {
      for (int ri = GEMDetId::minRingId; ri <= GEMDetId::maxRingId; ++ri) {
        for (int st = GEMDetId::minStationId; st <= GEMDetId::maxStationId; ++st) {
          for (int la = GEMDetId::minLayerId; la <= GEMDetId::maxLayerId; ++la) {
            for (int ch = 1 + GEMDetId::minChamberId; ch <= GEMDetId::maxChamberId; ++ch) {
              for (int ro = GEMDetId::minEtaPartitionId; ro <= GEMDetId::maxEtaPartitionId; ++ro) {
                GEMDetId id(re, ri, st, la, ch, ro);
                if ((id.region() != re) || (id.ring() != ri) || (id.station() != st) || (id.layer() != la) ||
                    (id.chamber() != ch) || (id.roll() != ro)) {
                  ++nerr;
                  std::cout << id << " should have been (" << re << ", " << ri << ", " << st << ", " << la << ", " << ch
                            << ", " << ro << ") " << std::hex << id.rawId() << std::dec << " ***** ERROR *****"
                            << std::endl;
                }
              }
            }
          }
        }
      }
    }
  }
  return nerr;
}

void testFail() {
  unsigned int raw1(671189248), raw2(688425248);
  GEMDetId id0(-1, 1, 1, 2, 10, 1);
  GEMDetId id1(raw1);
  GEMDetId id2(raw2);
  std::cout << " ID0: " << std::hex << id0.rawId() << std::dec << ":" << id0.rawId() << " " << id0
            << "\n ID1: " << std::hex << id1.rawId() << ":" << raw1 << std::dec << ":" << raw1 << " " << id1
            << "\n ID2: " << std::hex << id2.rawId() << ":" << raw2 << std::dec << ":" << raw2 << " " << id2
            << std::endl;
}

int main() {
  testFail();
  return testCell();
}
