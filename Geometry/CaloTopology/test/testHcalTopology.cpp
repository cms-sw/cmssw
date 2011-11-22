#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

int main(int argc, char* argv[]) {

  HcalTopology topology;
  for (int idet=0; idet<4; idet++) {
    HcalSubdetector subdet = HcalBarrel;
    if (idet == 1)      subdet = HcalOuter;
    else if (idet == 2) subdet = HcalEndcap;
    else if (idet == 3) subdet = HcalForward;
    for (int depth=1; depth<4; ++depth) {
      for (int ieta=-41; ieta<=41; ieta++) {
	for (int iphi=1; iphi<=72; iphi++) {
	  if (HcalDetId::validDetId(subdet,ieta,iphi,depth)) {
	    const HcalDetId id(subdet,ieta,iphi,depth);
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

  // test depth segmentation
  int e1[] = {1,2,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4,5,5};
  int e17[] = {1,1,1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4};
  std::vector<int> eta1(e1, e1+19);
  std::vector<int> eta17(e17, e17+22);
  topology.setDepthSegmentation(1, eta1);
  topology.setDepthSegmentation(17, eta17);
  for(int iring = 1; iring <=16; ++iring) {
    std::pair<int, int> bounds1 =  topology.segmentBoundaries(iring, 1);
    assert(bounds1.first == 0);
    assert(bounds1.second == 1);
    std::pair<int, int> bounds3 =  topology.segmentBoundaries(iring, 3);
    assert(bounds3.first == 5);
    assert(bounds3.second == 9);
    std::pair<int, int> bounds5 =  topology.segmentBoundaries(iring, 5);
    assert(bounds5.first == 17);
    assert(bounds5.second == 19); // past the end
  }
  for(int iring = 17; iring <=29; ++iring) {
    std::pair<int, int> bounds1 =  topology.segmentBoundaries(iring, 1);
    assert(bounds1.first == 0);
    assert(bounds1.second == 3);
    std::pair<int, int> bounds3 =  topology.segmentBoundaries(iring, 3);
    assert(bounds3.first == 7);
    assert(bounds3.second == 13);
    std::pair<int, int> bounds4 =  topology.segmentBoundaries(iring, 4);
    assert(bounds4.first == 13);
    assert(bounds4.second == 22);
  }
  return 0;
}
