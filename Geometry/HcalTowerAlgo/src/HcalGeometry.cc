#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

HcalGeometry::HcalGeometry() : lastReqDet_(DetId::Detector(0)), lastReqSubdet_(0) {
}
  

HcalGeometry::~HcalGeometry() {
  std::map<DetId, const CaloCellGeometry*>::iterator i;
  for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++)
    delete i->second;
  cellGeometries_.clear();
}


std::vector<DetId> HcalGeometry::getValidDetIds(DetId::Detector det, int subdet) const {
  if (lastReqDet_!=det || lastReqSubdet_!=subdet) {
    lastReqDet_=det;
    lastReqSubdet_=subdet;
    validIds_.clear();
  }
  if (validIds_.empty()) {
    std::map<DetId, const CaloCellGeometry*>::const_iterator i;    
    for (i=cellGeometries_.begin(); i!=cellGeometries_.end(); i++)
      if (i->first.det()==det && i->first.subdetId()==subdet) 
	 validIds_.push_back(i->first);
  }

  return validIds_;
}
