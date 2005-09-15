#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

namespace cms {



  CaloTowerDetId::CaloTowerDetId() { }

  CaloTowerDetId::CaloTowerDetId(uint32_t rawid) : HcalCompositeDetId(rawid) {
  }
  
  CaloTowerDetId::CaloTowerDetId(int tower_ieta, int tower_iphi) : HcalCompositeDetId(CaloTowerIdType, 0, tower_ieta, tower_iphi) {
  }

  CaloTowerDetId::CaloTowerDetId(const DetId& id) : HcalCompositeDetId(id) {
    if (getCompositeType()!=CaloTowerIdType) {
      throw new std::exception();
    }
  }

  CaloTowerDetId& CaloTowerDetId::operator=(const DetId& id) {
    HcalCompositeDetId::operator=(id);
    if (getCompositeType()!=CaloTowerIdType) {
      throw new std::exception();
    }
    return *this;
  }
}
