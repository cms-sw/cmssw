// Test the various geometry/position transformations in calorimeter
// regions input from the RCT to the GCT emulator.
// This used to be done in a class called L1GctMap.

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include <iostream>

using namespace std;

int main() {
  bool fail = false;

  // temp collection of IDs
  L1CaloRegionDetId ids[22][18];

  /// check conversion from RCT indices to ieta/iphi

  // construct HB/HE regions using RCT constructor
  for (unsigned crate = 0; crate < 18; crate++) {
    for (unsigned card = 0; card < 7; card++) {
      for (unsigned rgn = 0; rgn < 2; rgn++) {
        L1CaloRegionDetId r(crate, card, rgn);
        if ((r.rctCrate() != crate) || (r.rctCard() != card) || (r.rctRegion() != rgn)) {
          cout << "Error! : converting RCT indices to ieta/iphi at RCT crate " << crate << " card " << card
               << " region " << rgn << endl;
          fail = true;
        }
        if ((r.ieta() < 22) && (r.iphi() < 18)) {
          ids[r.ieta()][r.iphi()] = r;
        }
      }
    }
  }

  // construct HF regions using RCT constructor
  for (unsigned crate = 0; crate < 18; crate++) {
    for (unsigned rgn = 0; rgn < 8; rgn++) {
      L1CaloRegionDetId r(crate, 999, rgn);
      if ((r.rctCrate() != crate) || (r.rctRegion() != rgn)) {
        cerr << "Error! : RCT crate " << crate << " HF region " << rgn << endl;
        fail = true;
      }
      if ((r.ieta() < 22) && (r.iphi() < 18)) {
        ids[r.ieta()][r.iphi()] = r;
      }
    }
  }

  // check ieta/iphi map is filled correctly
  for (unsigned ieta = 0; ieta < 22; ieta++) {
    for (unsigned iphi = 0; iphi < 18; iphi++) {
      if (ids[ieta][iphi] != L1CaloRegionDetId(ieta, iphi)) {
        cerr << "Error! : missing L1CaloRegionDetId from RCT constructor at ieta=" << ieta << ", iphi=" << iphi << endl;
        fail = true;
      }
    }
  }

  /// check conversion from ieta/iphi to RCT indices

  for (unsigned ieta = 0; ieta < 22; ieta++) {
    for (unsigned iphi = 0; iphi < 18; iphi++) {
      L1CaloRegionDetId r(ieta, iphi);
      unsigned crate = r.rctCrate();
      unsigned card = r.rctCard();
      unsigned rgn = r.rctRegion();
      if (r != L1CaloRegionDetId(crate, card, rgn)) {
        cerr << "Error! : converting ieta/iphi to RCT indices at ieta=" << ieta << " iphi=" << iphi << endl;
        fail = true;
      }
    }
  }

  if (!fail)
    return 0;
  else
    return 1;
}
