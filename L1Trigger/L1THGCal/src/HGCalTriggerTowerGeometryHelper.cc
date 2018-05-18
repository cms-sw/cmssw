
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTowerGeometryHelper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include <cmath>
#include <iostream>
#include <fstream>


HGCalTriggerTowerGeometryHelper::HGCalTriggerTowerGeometryHelper(const edm::ParameterSet& conf) : type_(static_cast<HGCalTriggerTowerType>(conf.getParameter<int>("type"))) {


  if(type_ == HGCalTriggerTowerType::regular_xy_generic ||
    type_ == HGCalTriggerTowerType::regular_etaphi_generic ) {

      setRefCoordinates(conf.getParameter<double>("refCoord1"),
                        conf.getParameter<double>("refCoord2"),
                        conf.getParameter<double>("refZ"),
                        conf.getParameter<double>("binSizeCoord1"),
                        conf.getParameter<double>("binSizeCoord2"));

      // we read the TC to TriggerTower mapping
      std::ifstream l1tTriggerTowerMappingStream(conf.getParameter<edm::FileInPath>("L1TTriggerTowerMapping").fullPath());
      if(!l1tTriggerTowerMappingStream.is_open()) {
          throw cms::Exception("MissingDataFile")
              << "Cannot open HGCalTriggerGeometry L1TTriggerTowerMapping file\n";
      }

      unsigned trigger_cell_id = 0;
      unsigned short ix = 0;
      unsigned short iy = 0;

      for(; l1tTriggerTowerMappingStream >> trigger_cell_id >> ix >> iy;) {
        HGCalDetId detId(trigger_cell_id);
        int zside = detId.zside();
        l1t::HGCalTowerID towerId(zside, ix, iy);
        cells_to_trigger_towers_[trigger_cell_id] = towerId.rawId();
        GlobalPoint center = getPositionAtReferenceSurface(towerId);
        // std::cout << l1t::HGCalTowerID(towerId).zside() << " "
        //           << l1t::HGCalTowerID(towerId).coord1() << " "
        //           << l1t::HGCalTowerID(towerId).coord2()
        //           << "  eta: " << center.eta()
        //           << " phi: " << center.phi() << std::endl;
        tower_coords_.emplace_back(towerId.rawId(), center.eta(), center.phi());
      }
      l1tTriggerTowerMappingStream.close();

  } else if(type_ == HGCalTriggerTowerType::regular_etaphi) {
    // we create the binning using the config file
  }

}


// void HGCalTriggerTowerGeometryHelper::createTowerCoordinates(const std::vector<unsigned short>& tower_ids) {
//   if(tower_coords_.size() == 0) {
//     tower_coords_.reserve(tower_ids.size());
//     for (auto towerId : tower_ids) {
//       GlobalPoint center = getPositionAtReferenceSurface(l1t::HGCalTowerID(towerId));
//       // std::cout << l1t::HGCalTowerID(towerId).zside() << " "
//       //           << l1t::HGCalTowerID(towerId).coord1() << " "
//       //           << l1t::HGCalTowerID(towerId).coord2()
//       //           << "  eta: " << center.eta()
//       //           << " phi: " << center.phi() << std::endl;
//       tower_coords_.emplace_back(towerId, center.eta(), center.phi());
//     }
//   }
// }


const std::vector<l1t::HGCalTowerCoord>& HGCalTriggerTowerGeometryHelper::getTowerCoordinates() const {
  return tower_coords_;
}



unsigned short HGCalTriggerTowerGeometryHelper::getTriggerTowerFromTriggerCell(const unsigned trigger_cell_id) const {
  // FIXME: WE SHOULD IMPLEMENT A CHECK ON END?
  return cells_to_trigger_towers_.find(trigger_cell_id)->second;
}



GlobalPoint HGCalTriggerTowerGeometryHelper::getPositionAtReferenceSurface(const l1t::HGCalTowerID& towerId) const {
  GlobalPoint surface_center;
  if(type_ == HGCalTriggerTowerType::regular_xy_generic) {
    GlobalPoint referencePoint = towerId.zside() < 0 ?  GlobalPoint(refCoord1_, refCoord2_ , -1*referenceZ_) : GlobalPoint(refCoord1_, refCoord2_ , referenceZ_);
    surface_center = GlobalPoint(referencePoint.x()+towerId.coord1()*binSizeCoord1_,
                                 referencePoint.y()+towerId.coord2()*binSizeCoord2_,
                                 referencePoint.z());

  } else if(type_ == HGCalTriggerTowerType::regular_etaphi_generic) {
    float radius = fabs(referenceZ_/sinh(refCoord1_+towerId.coord1()*binSizeCoord1_));
    float phi = refCoord2_+towerId.coord2()*binSizeCoord2_;
    float zcoord = towerId.zside() < 0 ?  -1*referenceZ_ : referenceZ_;
    surface_center =  GlobalPoint(GlobalPoint::Cylindrical(radius, phi, zcoord));
  }
  return surface_center;

}
