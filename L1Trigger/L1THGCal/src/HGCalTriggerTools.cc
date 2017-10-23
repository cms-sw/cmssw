#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"


#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/EventSetup.h"


namespace {
  constexpr char hgcalee_sens[] = "HGCalEESensitive";
  constexpr char hgcalfh_sens[] = "HGCalHESiliconSensitive";

  constexpr std::float_t idx_to_thickness = std::float_t(100.0);

  template<typename DDD>
  inline void check_ddd(const DDD* ddd) {
    if( nullptr == ddd ) {
      throw cms::Exception("hgcal::HGCalTriggerTools")
        << "DDDConstants not accessibl to hgcal::HGCalTriggerTools!";
    }
  }

  template<typename GEOM>
  inline void check_geom(const GEOM* geom) {
    if( nullptr == geom ) {
      throw cms::Exception("hgcal::HGCalTriggerTools")
        << "Geometry not provided yet to hgcal::HGCalTriggerTools!";
    }
  }

  inline const HcalDDDRecConstants* get_ddd(const CaloSubdetectorGeometry* geom,
					    const HcalDetId& detid) {
    const HcalGeometry* hc = static_cast<const HcalGeometry*>(geom);
    const HcalDDDRecConstants* ddd = hc->topology().dddConstants();
    check_ddd(ddd);
    return ddd;
  }

  inline const HGCalDDDConstants* get_ddd(const CaloSubdetectorGeometry* geom,
					  const HGCalDetId& detid) {
    const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom);
    const HGCalDDDConstants* ddd = &(hg->topology().dddConstants());
    check_ddd(ddd);
    return ddd;
  }

}

void HGCalTriggerTools::setEventSetup(const edm::EventSetup& es) {

  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  es.get<CaloGeometryRecord>().get(triggerGeometry_);
  geom_ = triggerGeometry_.product();
  fhOffset_ = (geom_->eeTopology().dddConstants()).layers(true);
  bhOffset_ = fhOffset_ + (geom_->fhTopology().dddConstants()).layers(true);
}

GlobalPoint HGCalTriggerTools::getPosition(const DetId& id) const {
  auto geom = geom_->caloGeometry()->getSubdetectorGeometry(id);
  check_geom(geom);
  GlobalPoint position;
  if( id.det() == DetId::Hcal ) {
    position = geom->getGeometry(id)->getPosition();
  } else {
    const auto* hg = static_cast<const HGCalGeometry*>(geom);
    position = hg->getPosition(id);
  }
  return position;
}


// int HGCalTriggerTools::zside(const DetId& id) const {
//   int zside = 0;
//   if( id.det() == DetId::Forward) {
//     const HGCalDetId hid(id);
//     zside = hid.zside();
//   } else if( id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
//     const HcalDetId hcid(id);
//     zside = hcid.zside();
//   }
//   return zside;
// }
//
// std::float_t HGCalTriggerTools::getSiThickness(const DetId& id) const {
//   auto geom = geom_->getSubdetectorGeometry(id);
//   check_geom(geom);
//   if( id.det() != DetId::Forward ) {
//     LogDebug("getSiThickness::InvalidSiliconDetid")
//       << "det id: " << id.rawId() << " is not HGCal silicon!";
//   }
//   const HGCalDetId hid(id);
//   auto ddd = get_ddd(geom,hid);
//   unsigned int wafer = hid.wafer();
//   int tidx = ddd->waferTypeL(wafer);
//   return idx_to_thickness*tidx;
// }
//
// std::float_t HGCalTriggerTools::getRadiusToSide(const DetId& id) const {
//   auto geom = geom_->getSubdetectorGeometry(id);
//   check_geom(geom);
//   if( id.det() != DetId::Forward ) {
//     edm::LogError("getRadiusToSide::InvalidSiliconDetid")
//       << "det id: " << id.rawId() << " is not HGCal silicon!";
//     return std::numeric_limits<std::float_t>::max();
//   }
//   const HGCalDetId hid(id);
//   auto ddd = get_ddd(geom,hid);
//   std::float_t size = ddd->cellSizeHex(hid.waferType());
//   return size;
// }
//
// unsigned int HGCalTriggerTools::getLayer(const ForwardSubdetector type) const {
//
//   int layer;
//   switch (type) {
//     case(ForwardSubdetector::HGCEE): {
//       auto geomEE = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCEE));
//       layer       = (geomEE->topology().dddConstants()).layers(true);
//       break;
//     }
//     case (ForwardSubdetector::HGCHEF): {
//       auto geomFH = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCHEF));
//       layer       = (geomFH->topology().dddConstants()).layers(true);
//       break;
//     }
//     case (ForwardSubdetector::HGCHEB): {
//       auto geomBH = static_cast<const HcalGeometry*>(geom_->getSubdetectorGeometry(DetId::Hcal,HcalSubdetector::HcalEndcap));
//       layer       = (geomBH->topology().dddConstants())->getMaxDepth(1);
//       break;
//     }
//     case (ForwardSubdetector::ForwardEmpty): {
//       auto geomEE = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCEE));
//       layer       = (geomEE->topology().dddConstants()).layers(true);
//       auto geomFH = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCHEF));
//       layer      += (geomFH->topology().dddConstants()).layers(true);
//       auto geomBH = static_cast<const HcalGeometry*>(geom_->getSubdetectorGeometry(DetId::Hcal,HcalSubdetector::HcalEndcap));
//       layer      += (geomBH->topology().dddConstants())->getMaxDepth(1);
//       break;
//     }
//     default: layer = 0;
//   }
//   return (unsigned int)(layer);
// }

unsigned int HGCalTriggerTools::getLayer(const DetId& id) const {
  unsigned int layer = std::numeric_limits<unsigned int>::max();
  if( id.det() == DetId::Forward) {
    const HGCalDetId hid(id);
    layer = hid.layer();
  } else if( id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
    const HcalDetId hcid(id);
    layer = hcid.depth();
  }
  return layer;
}

unsigned int HGCalTriggerTools::getLayerWithOffset(const DetId& id) const {
  unsigned int layer = getLayer(id);
  if( id.det() == DetId::Forward && id.subdetId() == HGCHEF ) {
    layer += fhOffset_;
  } else if( (id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) ||
             (id.det() == DetId::Forward && id.subdetId() == HGCHEB) ) {
    layer += bhOffset_;
  }
  return layer;
}

// unsigned int HGCalTriggerTools::getWafer(const DetId& id) const {
//   if( id.det() != DetId::Forward ) {
//     edm::LogError("getWafer::InvalidSiliconDetid")
//       << "det id: " << id.rawId() << " is not HGCal silicon!";
//     return std::numeric_limits<unsigned int>::max();
//   }
//   const HGCalDetId hid(id);
//   unsigned int wafer = hid.wafer();
//   return wafer;
// }
//
// unsigned int HGCalTriggerTools::getCell(const DetId& id) const {
//   if( id.det() != DetId::Forward ) {
//     edm::LogError("getCell::InvalidSiliconDetid")
//       << "det id: " << id.rawId() << " is not HGCal silicon!";
//     return std::numeric_limits<unsigned int>::max();
//   }
//   const HGCalDetId hid(id);
//   unsigned int cell = hid.cell();
//   return cell;
// }
//
// bool HGCalTriggerTools::isHalfCell(const DetId& id) const {
//   if( id.det() != DetId::Forward ) {
//     return false;
//   }
//   auto geom = geom_->getSubdetectorGeometry(id);
//   check_geom(geom);
//   const HGCalDetId hid(id);
//   auto ddd = get_ddd(geom,hid);
//   const int waferType = ddd->waferTypeT(hid.waferType());
//   return ddd->isHalfCell(waferType,hid.cell());
// }

float HGCalTriggerTools::getEta(const GlobalPoint& position, const float& vertex_z) const {
  GlobalPoint corrected_position = GlobalPoint(position.x(), position.y(), position.z()-vertex_z);
  return corrected_position.eta();
}

float HGCalTriggerTools::getEta(const DetId& id, const float& vertex_z) const {
  GlobalPoint position = getPosition(id);
  float eta = getEta(position, vertex_z);
  return eta;
}

float HGCalTriggerTools::getPhi(const GlobalPoint& position) const {
  float phi = atan2(position.y(),position.x());
  return phi;
}

float HGCalTriggerTools::getPhi(const DetId& id) const {
  GlobalPoint position = getPosition(id);
  float phi = atan2(position.y(),position.x());
  return phi;
}

float HGCalTriggerTools::getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z) const {
  float eta = getEta(position, vertex_z);
  float pt = hitEnergy / cosh(eta);
  return pt;
}

float HGCalTriggerTools::getPt(const DetId& id, const float& hitEnergy, const float& vertex_z) const {
  GlobalPoint position = getPosition(id);
  float eta = getEta(position, vertex_z);
  float pt = hitEnergy / cosh(eta);
  return pt;
}
