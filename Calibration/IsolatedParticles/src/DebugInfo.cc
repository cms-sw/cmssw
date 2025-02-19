#include "Calibration/IsolatedParticles/interface/DebugInfo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include<iostream>

namespace spr{

  void debugEcalDets(unsigned int i, const DetId& det, bool flag) {

    std::cout << "Cell [" << i << "] 0x";
    if (det.subdetId() == EcalBarrel) {
      EBDetId id = det;
      std::cout << std::hex << det() << std::dec << " " << id;
    } else if (det.subdetId() == EcalEndcap) {
      EEDetId id = det;
      std::cout << std::hex << det() << std::dec << " " << id;
    } else {
      std::cout << std::hex << det() << std::dec << " Unknown Type";
    }
    if (flag) std::cout << std::endl;
  }
 
  void debugEcalDets(unsigned int last, std::vector<DetId>& vdets) {

     for (unsigned int i=last; i<vdets.size(); ++i) {
       debugEcalDets (i, vdets[i], true);
     }
  }

  void debugEcalDets(unsigned int last, std::vector<DetId>& vdets, 
		     std::vector<CaloDirection>& dirs) {

    for (unsigned int i=last; i<vdets.size(); ++i) {
      debugEcalDets (i, vdets[i], false);
      std::cout << " along " << dirs[i] << std::endl;
    }
  }

  void debugHcalDets(unsigned int last, std::vector<DetId>& vdets) {

    for (unsigned int i=last; i<vdets.size(); ++i) {
      HcalDetId id = vdets[i]();
      std::cout << "Cell [" << i << "] 0x" << std::hex << vdets[i]() 
		<< std::dec << " " << id << std::endl;
    }
  }
}
