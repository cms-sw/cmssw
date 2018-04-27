
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTowerGeometryHelper.h"
#include <cmath>
#include <iostream>

void HGCalTriggerTowerGeometryHelper::createTowerCoordinates(const std::vector<unsigned short>& tower_ids) {
  if(coord.size() == 0) {
    coord.reserve(tower_ids.size());
    for (auto towerId : tower_ids) {
      GlobalPoint center = getPositionAtReferenceSurface(l1t::HGCalTowerID(towerId));
      std::cout << l1t::HGCalTowerID(towerId).zside() << " "
                << l1t::HGCalTowerID(towerId).iX() << " "
                << l1t::HGCalTowerID(towerId).iY()
                << "  eta: " << center.eta()
                << " phi: " << center.phi() << std::endl;
      coord.emplace_back(towerId, center.eta(), center.phi());
    }
  }
}


const std::vector<l1t::HGCalTowerCoord>& HGCalTriggerTowerGeometryHelper::getTowerCoordinates() const {
  return coord;
}





GlobalPoint HGCalTriggerTowerGeometryHelper::getPositionAtReferenceSurface(const l1t::HGCalTowerID& towerId) const {
  GlobalPoint surface_center;
  if(type_ == HGCalTriggerTowerType::regular_xy) {
    GlobalPoint referencePoint = towerId.zside() < 0 ?  GlobalPoint(refCoord1_, refCoord2_ , -1*referenceZ_) : GlobalPoint(refCoord1_, refCoord2_ , referenceZ_);
    surface_center = GlobalPoint(referencePoint.x()+towerId.iX()*binSizeCoord1_,
                                 referencePoint.y()+towerId.iY()*binSizeCoord2_,
                                 referencePoint.z());

  } else if(type_ == HGCalTriggerTowerType::regular_etaphi) {
    float radius = fabs(referenceZ_/sinh(refCoord1_+towerId.iX()*binSizeCoord1_));
    float phi = refCoord2_+towerId.iY()*binSizeCoord2_;
    float zcoord = towerId.zside() < 0 ?  -1*referenceZ_ : referenceZ_;
    surface_center =  GlobalPoint(GlobalPoint::Cylindrical(radius, phi, zcoord));
  }
  return surface_center;

}
