#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

CaloTowerTopology::CaloTowerTopology() {
}

CaloTowerDetId CaloTowerTopology::towerOf(const DetId& id) const {
  CaloTowerDetId tid; // null to start with

  if (id.det()==DetId::Hcal) { 
    HcalDetId hid(id);
    if (hid.subdet()==HcalForward && hid.ietaAbs()==29) { // special handling for tower 29
      tid=CaloTowerDetId(30*hid.zside(),hid.iphi());
    } else {
      tid=CaloTowerDetId(hid.ieta(),hid.iphi());
    }
  }

  return tid;
}

std::vector<DetId> CaloTowerTopology::constituentsOf(const CaloTowerDetId& id) const {
  std::vector<DetId> items;

  // dealing with topo dependency...
  
  return items;
}
