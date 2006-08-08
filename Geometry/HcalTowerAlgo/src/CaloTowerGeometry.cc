#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"

CaloTowerGeometry::CaloTowerGeometry() {
}
  

CaloTowerGeometry::~CaloTowerGeometry() {
  std::map<DetId, const CaloCellGeometry*>::iterator i;
  for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++)
    delete i->second;
  cellGeometries_.clear();
}

