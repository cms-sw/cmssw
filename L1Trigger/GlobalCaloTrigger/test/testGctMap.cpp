
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
    



  return 0;

}
