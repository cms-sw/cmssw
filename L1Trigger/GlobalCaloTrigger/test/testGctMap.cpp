
// Test the various geometry/position transformations in calorimeter
// regions input from the RCT to the GCT emulator.
// This used to be done in a class called L1GctMap.

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include <iostream>

using namespace std;

int main() {

  bool testPass=true;

  cout << "\nTesting eta,phi to/from ID conversion" << endl << endl;

  for (unsigned ieta=0; ieta<22; ieta++) {   // loop over eta

    cout << ieta << " ";

    for (unsigned iphi=0; iphi<18; iphi++) {   // loop over phi

      L1CaloRegionDetId temp(ieta, iphi);

      if ( temp.iphi() != iphi ) { testPass = false;
	cout << "Error : phi->id->phi conversion failed for phi=" << iphi << " stored phi=" << temp.iphi() << endl;
      }

      if ( temp.ieta() != ieta ) { testPass = false;
	cout << "Error : eta->id->eta conversion failed for eta=" << ieta << " stored eta=" << temp.ieta() << endl;
      }

    }

  }

  cout << endl;


  cout << "\nTest of L1GctMap " << (testPass ? "passed!" : "failed!") << endl;
  return 0;

}
