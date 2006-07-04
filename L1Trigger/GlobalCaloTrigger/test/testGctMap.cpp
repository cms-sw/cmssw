
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

  cout << "\nTesting source card mapping\n";

  unsigned crate=0;
  unsigned card=0;
  unsigned phi=4;
  for (; crate<9; crate++) {
    ++card;
    cout << "Crate " << crate << " source 2 \n"; 
    unsigned eta;
    for (unsigned in=0; in<12; in++) {
      unsigned phj;
      switch (in) {
      case 0 : eta = 6; if (phi==0) { phj=17; } else { phj=phi-1; } break;
      case 1 : eta = 5; if (phi==0) { phj=17; } else { phj=phi-1; } break;
      case 2 : eta = 4; phj=phi; break;
      case 3 : eta = 4; if (phi==0) { phj=17; } else { phj=phi-1; } break;
      default : eta = 3-((in-4)%4); phj = ((phi+18-((in-4)/4))%18); break;
      }
      L1CaloRegion temp(100,
			false, false, false, false,
			eta, phj);
      if ((temp.rctCrate()!=crate) ||
	  (temp.gctCard()!=card) ||
	  (temp.gctRegionIndex()!=in)) { testPass = false;
	cout << "Error for crate " << crate
	     << " source card " << card << ", input " << in
	     << " id is " << "  (" << temp.gctEta() << "," << temp.gctPhi() << ")" << endl;
	cout << "rctCrate "    << temp.rctCrate()
	     << " sourceCard " << temp.gctCard()
	     << " scOutput "   << temp.gctRegionIndex() << endl;
      } else { cout << "(" << temp.gctEta() << "," << temp.gctPhi() << ") "; }
    }
    cout << endl;

    cout << "Crate " << crate << " source 3 \n";
    ++card;
    eta = 11;
    for (unsigned in=0; in<10; in++) {
      --eta;
      L1CaloRegion temp(100,
			false, false, false, false,
			eta, phi);
      if ((temp.rctCrate()!=crate) ||
	  (temp.gctCard()!=card) ||
	  (temp.gctRegionIndex()!=in)) { testPass = false;
	cout << "Error for crate " << crate
	     << " source card " << card << ", input " << in
	     << " id is " << "  (" << temp.gctEta() << "," << temp.gctPhi() << ")" << endl;
	cout << "rctCrate "    << temp.rctCrate()
	     << " sourceCard " << temp.gctCard()
	     << " scOutput "   << temp.gctRegionIndex() << endl;
      } else { cout << "(" << temp.gctEta() << "," << temp.gctPhi() << ") "; }
      if (eta == 5) {
	eta = 11;
	if (phi == 0) { phi=18; }
	--phi;
      }
    }
    cout << endl;
    ++card;
    if (phi==0) { phi=18; }
    --phi;
  }

  for (; crate<18; crate++) {
    ++card;
    cout << "Crate " << crate << " source 2 \n"; 
    unsigned eta;
    for (unsigned in=0; in<12; in++) {
      unsigned phj;
      switch (in) {
      case 0 : eta = 15; if (phi==0) { phj=17; } else { phj=phi-1; } break;
      case 1 : eta = 16; if (phi==0) { phj=17; } else { phj=phi-1; } break;
      case 2 : eta = 17; phj=phi; break;
      case 3 : eta = 17; if (phi==0) { phj=17; } else { phj=phi-1; } break;
      default : eta = 18+((in-4)%4); phj = ((phi+18-((in-4)/4))%18); break;
      }
      L1CaloRegion temp(100,
			false, false, false, false,
			eta, phj);
      if ((temp.rctCrate()!=crate) ||
	  (temp.gctCard()!=card) ||
	  (temp.gctRegionIndex()!=in)) { testPass = false;
	cout << "Error for crate " << crate
	     << " source card " << card << ", input " << in
	     << " id is " << "  (" << temp.gctEta() << "," << temp.gctPhi() << ")" << endl;
	cout << "rctCrate "    << temp.rctCrate()
	     << " sourceCard " << temp.gctCard()
	     << " scOutput "   << temp.gctRegionIndex() << endl;
      } else { cout << "(" << temp.gctEta() << "," << temp.gctPhi() << ") "; }
    }
    cout << endl;

    cout << "Crate " << crate << " source 3 \n";
    ++card;
    eta = 11;
    for (unsigned in=0; in<10; in++) {
      L1CaloRegion temp(100,
			false, false, false, false,
			eta, phi);
      if ((temp.rctCrate()!=crate) ||
	  (temp.gctCard()!=card) ||
	  (temp.gctRegionIndex()!=in)) { testPass = false;
	cout << "Error for crate " << crate
	     << " source card " << card << ", input " << in
	     << " id is " << "  (" << temp.gctEta() << "," << temp.gctPhi() << ")" << endl;
	cout << "rctCrate "    << temp.rctCrate()
	     << " sourceCard " << temp.gctCard()
	     << " scOutput "   << temp.gctRegionIndex() << endl;
      } else { cout << "(" << temp.gctEta() << "," << temp.gctPhi() << ") "; }
      if (eta == 16) {
	eta = 10;
	if (phi == 0) { phi=18; }
	--phi;
      }
      ++eta;
    }
    cout << endl;
    ++card;
    if (phi==0) { phi=18; }
    --phi;
  }

  cout << "\nTest of L1GctMap " << (testPass ? "passed!" : "failed!") << endl;
  return 0;

}
