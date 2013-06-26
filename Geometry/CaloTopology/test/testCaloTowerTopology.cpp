#include <iostream>
#include <string>
#include <vector>

#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

int main(int /*argc*/, char** /*argv*/) {

  CaloTowerTopology topology;
  for (int ieta=-41; ieta<=41; ieta++) {
    for (int iphi=1; iphi<=72; iphi++) {
      if (CaloTowerDetId::validDetId(ieta,iphi)) {
	const CaloTowerDetId id(ieta,iphi);
	std::vector<DetId> idE = topology.east(id);
	std::vector<DetId> idW = topology.west(id);
	std::vector<DetId> idN = topology.north(id);
	std::vector<DetId> idS = topology.south(id);
	std::cout << "Neighbours for : Tower " << id << std::endl;
	std::cout << "          " << idE.size() << " sets along East:";
	for (unsigned int i=0; i<idE.size(); ++i) 
	  std::cout << " " << (CaloTowerDetId)(idE[i]());
	std::cout << std::endl;
	std::cout << "          " << idW.size() << " sets along West:";
	for (unsigned int i=0; i<idW.size(); ++i) 
	  std::cout << " " << (CaloTowerDetId)(idW[i]());
	std::cout << std::endl;
	std::cout << "          " << idN.size() << " sets along North:";
	for (unsigned int i=0; i<idN.size(); ++i) 
	  std::cout << " " << (CaloTowerDetId)(idN[i]());
	std::cout << std::endl;
	std::cout << "          " << idS.size() << " sets along South:";
	for (unsigned int i=0; i<idS.size(); ++i) 
	  std::cout << " " << (CaloTowerDetId)(idS[i]());
	std::cout << std::endl;
      }
    }
  }
  return 0;
}
