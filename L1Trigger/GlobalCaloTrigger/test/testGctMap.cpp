
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctRegion.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctMap.h"

#include <iostream>

using namespace std;

int main() {

  L1GctMap* map = L1GctMap::getMap();

  cout << "Testing eta,phi to/from ID conversion" << endl << endl;

  for (unsigned ieta=0; ieta<22; ieta++) {   // loop over eta

    cout << ieta << " ";

    for (unsigned iphi=0; iphi<18; iphi++) {   // loop over phi

      if ( map->phi(map->id(ieta, iphi)) != iphi ) {
	cout << "Error : phi->id->phi conversion failed for phi=" << iphi << " id=" << map->id(ieta,iphi) << endl;
      }

      if ( map->eta(map->id(ieta, iphi)) != ieta ) {
	cout << "Error : eta->id->eta conversion failed for eta=" << iphi << " id=" << map->id(ieta,iphi) << endl;
      }

      cout << map->id(ieta, iphi) << " ";
      cout << map->eta(map->id(ieta, iphi)) << " ";
      cout << map->phi(map->id(ieta, iphi)) << " ";      

    }

    cout << endl;

  }

  cout << "testing source card mapping\n";

  for (unsigned crate=0; crate<18; crate++) {

    cout << "Crate " << crate << " source 2 \n"; 
    for (unsigned in=0; in<12; in++) {
      L1GctRegion temp(map->id(crate,2,in), 100,
		       false, false, false, false);
      if ((map->rctCrate(temp)!=crate) ||
	  (map->sourceCardType(temp)!=2) ||
	  (map->sourceCardOutput(temp)!=in)) {
	cout << "Error for crate " << crate
	     << " source card 2, input " << in
	     << " id is " << temp.id() << endl;
      } else { cout << "  " << temp.id(); }
    }
    cout << endl;

    cout << "Crate " << crate << " source 3 \n";
    for (unsigned in=0; in<10; in++) {
      L1GctRegion temp(map->id(crate,3,in), 100,
		       false, false, false, false);
      if ((map->rctCrate(temp)!=crate) ||
	  (map->sourceCardType(temp)!=3) ||
	  (map->sourceCardOutput(temp)!=in)) {
	cout << "Error for crate " << crate
	     << " source card 3, input " << in
	     << " id is " << temp.id() << endl;
      } else { cout << "  " << temp.id(); }
    }
    cout << endl;
  }

  return 0;

}
