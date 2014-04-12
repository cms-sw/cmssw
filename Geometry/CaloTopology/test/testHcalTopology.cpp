#include <iostream>
#include <string>
#include <vector>

#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

int main(int /*argc*/, char** /*argv*/) {

  // FIXME: for SLHC
  HcalTopologyMode::Mode mode = HcalTopologyMode::LHC;
  int maxDepthHB = 2;
  int maxDepthHE = 3;
  
  HcalTopology topology(mode, maxDepthHB, maxDepthHE);
  for (int idet=0; idet<4; idet++) {
    HcalSubdetector subdet = HcalBarrel;
    if (idet == 1)      subdet = HcalOuter;
    else if (idet == 2) subdet = HcalEndcap;
    else if (idet == 3) subdet = HcalForward;
    for (int depth=1; depth<4; ++depth) {
      for (int ieta=-41; ieta<=41; ieta++) {
	for (int iphi=1; iphi<=72; iphi++) {
	  const HcalDetId id(subdet,ieta,iphi,depth);
	  if (topology.valid(id)) {
	    std::vector<DetId> idE = topology.east(id);
	    std::vector<DetId> idW = topology.west(id);
	    std::vector<DetId> idN = topology.north(id);
	    std::vector<DetId> idS = topology.south(id);
	    std::vector<DetId> idU = topology.up(id);
	    std::cout << "Neighbours for : Tower " << id << std::endl;
	    std::cout << "          " << idE.size() << " sets along East:";
	    for (unsigned int i=0; i<idE.size(); ++i) 
	      std::cout << " " << (HcalDetId)(idE[i]());
	    std::cout << std::endl;
	    std::cout << "          " << idW.size() << " sets along West:";
	    for (unsigned int i=0; i<idW.size(); ++i) 
	      std::cout << " " << (HcalDetId)(idW[i]());
	    std::cout << std::endl;
	    std::cout << "          " << idN.size() << " sets along North:";
	    for (unsigned int i=0; i<idN.size(); ++i) 
	      std::cout << " " << (HcalDetId)(idN[i]());
	    std::cout << std::endl;
	    std::cout << "          " << idS.size() << " sets along South:";
	    for (unsigned int i=0; i<idS.size(); ++i) 
	      std::cout << " " << (HcalDetId)(idS[i]());
	    std::cout << std::endl;
	    std::cout << "          " << idU.size() << " sets up in depth:";
	    for (unsigned int i=0; i<idU.size(); ++i) 
	      std::cout << " " << (HcalDetId)(idU[i]());
	    std::cout << std::endl;
	  }
	}
      }
    }
  }
  return 0;
}
