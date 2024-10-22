// Test the various geometry/position transformations in
// candidates from the GCT
//
// Alex Tapper 5th October 2007
//

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include <iostream>

using namespace std;

int main() {
  bool fail = false;

  // Check we get back out what we constructed the object with even when we use the regionID object
  for (unsigned rank = 1; rank < 64;
       rank++) {  // Start at rank=1 since rank =0 returns the default values for the hardware, not what you input for eta and phi
    for (unsigned phi = 0; phi < 18; phi++) {
      for (unsigned eta = 0; eta < 7; eta++) {
        for (unsigned etaSign = 0; etaSign < 2; etaSign++) {
          for (unsigned iso = 0; iso < 2; iso++) {
            unsigned gctEta = (etaSign << 3) | (eta & 0x7);
            L1GctEmCand e(rank, phi, gctEta, iso);
            if (e.rank() != rank || e.etaIndex() != gctEta || e.phiIndex() != phi || e.isolated() != iso) {
              cout << "Error in L1GctEmCand constructor rank=" << rank << " phi=" << phi << " eta=" << gctEta
                   << " iso=" << iso << endl;
              fail = true;
            }
            unsigned ieta = etaSign ? 10 - eta : 11 + eta;
            if (e.regionId().iphi() != phi || e.regionId().ieta() != ieta) {
              cout << "Error in L1GctEmCand regionId() eta=" << ieta << " phi=" << phi << endl;
              fail = true;
            }
          }
        }
      }
    }
  }

  // Check we get back out what we constructed the object with even when we use the regionID object
  for (unsigned rank = 0; rank < 64; rank++) {
    for (unsigned phi = 0; phi < 18; phi++) {
      for (unsigned eta = 0; eta < 7; eta++) {
        for (unsigned tau = 0; tau < 2; tau++) {
          for (unsigned etaSign = 0; etaSign < 2; etaSign++) {
            unsigned gctEta = (etaSign << 3) | (eta & 0x7);
            L1GctJetCand j(rank, phi, gctEta, tau, false);
            if (j.rank() != rank || j.etaIndex() != gctEta || j.phiIndex() != phi || j.isTau() != tau) {
              cout << "Error in L1GctJetCand constructor rank=" << rank << " phi=" << phi << " eta=" << gctEta
                   << " tau=" << tau << endl;
              fail = true;
            }
            unsigned ieta = etaSign ? 10 - eta : 11 + eta;
            if (j.regionId().iphi() != phi || j.regionId().ieta() != ieta) {
              cout << "Error in L1GctJetCand regionId() eta=" << ieta << " phi=" << phi << endl;
              fail = true;
            }
          }
        }
      }
    }
  }

  // Check we get back out what we constructed the object with even when we use the regionID object
  for (unsigned rank = 0; rank < 64; rank++) {
    for (unsigned phi = 0; phi < 18; phi++) {
      for (unsigned eta = 0; eta < 3; eta++) {
        for (unsigned etaSign = 0; etaSign < 2; etaSign++) {
          unsigned gctEta = (etaSign << 3) | (eta & 0x7);
          L1GctJetCand j(rank, phi, gctEta, false, true);
          if (j.rank() != rank || j.etaIndex() != gctEta || j.phiIndex() != phi) {
            cout << "Error in L1GctJetCand (forward) constructor rank=" << rank << " phi=" << phi << " eta=" << gctEta
                 << endl;
            fail = true;
          }
          unsigned ieta = etaSign ? 3 - eta : 18 + eta;
          if (j.regionId().iphi() != phi || j.regionId().ieta() != ieta) {
            cout << "Error in L1GctJetCand (forward) regionId() eta=" << ieta << " phi=" << phi << endl;
            fail = true;
          }
        }
      }
    }
  }

  if (!fail)
    return 0;
  else
    return 1;
}
