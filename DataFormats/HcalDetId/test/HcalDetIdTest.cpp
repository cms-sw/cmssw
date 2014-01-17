#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <string>
#include <iostream>

void testDetId(HcalSubdetector subdet, std::string str) {

  int etamin, etamax, dmin, dmax;
  switch(subdet) {
  case HcalEndcap:
    dmin = 1; dmax = 3; etamin = 17; etamax = 29;
    break;
  case HcalForward:
    dmin = 1; dmax = 2; etamin = 29; etamax = 41; 
    break;
  case HcalOuter:
    dmin = 4; dmax = 4; etamin = 1;  etamax = 15;
    break;
  default:
    dmin = 1; dmax = 2; etamin = 1;  etamax = 16;
    break;
  }
  int phis[4] = {11, 27, 47, 63};
  int zside[2] = {1, -1};
  std::string sp[2] = {"  ", " "};

  std::cout << std::endl << "HCAL Det ID for " << str << " (" << subdet 
	    << ")" << std::endl;
  for (int eta=etamin; eta <= etamax; ++eta) {
    for (int depth=dmin; depth <= dmax; ++depth) {
      for (int fi=0; fi<4; ++fi) {
	for (int iz=0; iz<2; ++iz) {
	  HcalDetId id1 = HcalDetId(subdet, zside[iz]*eta, phis[fi], depth);
	  HcalDetId id2 = HcalDetId(subdet, zside[iz]*eta, phis[fi], depth, true);
	  std::cout << "Input " << subdet << ":" << zside[iz]*eta << ":" 
		    << phis[fi] << ":" << depth << sp[iz] << " New " << id1 
		    << sp[iz] << " Old " << id2 << std::endl;
	}
      }
    }
  }
  std::cout << std::endl << std::endl;
}

int main() {

  std::cout << "Test Hcal DetID in old and new Format" << std::endl;

  testDetId (HcalBarrel,  "BARREL"  );
  testDetId (HcalEndcap,  "ENDCAP"  );
  testDetId (HcalOuter,   "OUTER "  );
  testDetId (HcalForward, "FORWARD" );
    
  return 0;
}
