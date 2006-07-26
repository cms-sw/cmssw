#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

CaloTowerGeometry::CaloTowerGeometry() {
}
  

CaloTowerGeometry::~CaloTowerGeometry() {
  std::map<DetId, const CaloCellGeometry*>::iterator i;
  for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++)
    delete i->second;
  cellGeometries_.clear();
}

bool CaloTowerGeometry::present(const DetId& id) const {
  return CaloSubdetectorGeometry::present(CaloTowerDetId(id)); // handle conversion if needed
}

const CaloCellGeometry* CaloTowerGeometry::getGeometry(const DetId& id) const {
  return CaloSubdetectorGeometry::getGeometry(CaloTowerDetId(id)); // handle conversion if needed
}
