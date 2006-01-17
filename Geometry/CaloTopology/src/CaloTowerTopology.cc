#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

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
  } else if (id.det()==DetId::Ecal) {
    EcalSubdetector esd=(EcalSubdetector)id.subdetId();
    if (esd==EcalBarrel) {
      EBDetId ebid(id);
      tid=CaloTowerDetId(ebid.tower_ieta(),ebid.tower_iphi());
    }
  }

  return tid;
}

std::vector<DetId> CaloTowerTopology::constituentsOf(const CaloTowerDetId& id) const {
  std::vector<DetId> items;

  // dealing with topo dependency...
  
  return items;
}
