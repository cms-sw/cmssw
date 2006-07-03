
// Test the various geometry/position transformations in calorimeter
// regions input from the RCT to the GCT emulator.
// This used to be done in a class called L1GctMap.

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include <iostream>

using namespace std;

int main() {

  bool testPass=true;

  cout << "Testing eta,phi to/from ID conversion" << endl << endl;

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

  cout << "testing source card mapping\n";

//   for (unsigned crate=0; crate<18; crate++) {

//     cout << "Crate " << crate << " source 2 \n"; 
//     for (unsigned in=0; in<12; in++) {
//       L1CaloRegion temp(map->id(crate,2,in), 100,
// 			false, false, false, false);
//       if ((map->rctCrate(temp)!=crate) ||
// 	  (map->sourceCardType(temp)!=2) ||
// 	  (map->sourceCardOutput(temp)!=in)) { testPass = false;
// 	cout << "Error for crate " << crate
// 	     << " source card 2, input " << in
// 	     << " id is " << temp.id() << endl;
// 	cout << "rctCrate " << map->rctCrate(temp)
// 	     << " scType " << map->sourceCardType(temp)
// 	     << " scOutput " << map->sourceCardOutput(temp) << endl;
//       } else { cout << "  " << temp.id(); }
//     }
//     cout << endl;

//     cout << "Crate " << crate << " source 3 \n";
//     for (unsigned in=0; in<10; in++) {
//       L1CaloRegion temp(map->id(crate,3,in), 100,
// 			false, false, false, false);
//       if ((map->rctCrate(temp)!=crate) ||
// 	  (map->sourceCardType(temp)!=3) ||
// 	  (map->sourceCardOutput(temp)!=in)) { testPass = false;
// 	cout << "Error for crate " << crate
// 	     << " source card 3, input " << in
// 	     << " id is " << temp.id() << endl;
// 	cout << "rctCrate " << map->rctCrate(temp)
// 	     << " scType " << map->sourceCardType(temp)
// 	     << " scOutput " << map->sourceCardOutput(temp) << endl;
//       } else { cout << "  " << temp.id(); }
//     }
//     cout << endl;
//   }

  cout << "\nTest of L1GctMap " << (testPass ? "passed!" : "failed!") << endl;
  return 0;

}
