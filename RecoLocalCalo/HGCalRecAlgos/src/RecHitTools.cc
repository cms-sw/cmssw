#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace hgcal;

namespace {
  constexpr char hgcalee_sens[] = "HGCalEESensitive";
  constexpr char hgcalfh_sens[] = "HGCalHESiliconSensitive";

  constexpr std::float_t idx_to_thickness = std::float_t(100.0);

  template<typename DDD>
  inline void check_ddd(const DDD* ddd) {
    if( nullptr == ddd ) {
      throw cms::Exception("hgcal::RecHitTools")
        << "DDDConstants not accessibl to hgcal::RecHitTools!";
    }
  }

  template<typename GEOM>
  inline void check_geom(const GEOM* geom) {
    if( nullptr == geom ) {
      throw cms::Exception("hgcal::RecHitTools")
        << "Geometry not provided yet to hgcal::RecHitTools!";
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

  inline const HGCalDDDConstants* get_ddd(const CaloSubdetectorGeometry* geom,
					  const HGCSiliconDetId& detid) {
    const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom);
    const HGCalDDDConstants* ddd = &(hg->topology().dddConstants());
    check_ddd(ddd);
    return ddd;
  }

  inline const HGCalDDDConstants* get_ddd(const CaloSubdetectorGeometry* geom,
					  const HGCScintillatorDetId& detid) {
    const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom);
    const HGCalDDDConstants* ddd = &(hg->topology().dddConstants());
    check_ddd(ddd);
    return ddd;
  }

}

void RecHitTools::getEvent(const edm::Event& ev) {
}

void RecHitTools::getEventSetup(const edm::EventSetup& es) {

  edm::ESHandle<CaloGeometry> geom;
  es.get<CaloGeometryRecord>().get(geom);

  geom_ = geom.product();
  unsigned int wmaxEE(0), wmaxFH(0);
  auto geomEE = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::HGCalEE,ForwardSubdetector::ForwardEmpty));
  //check if it's the new geometry
  if(geomEE) {
    geometryType_ = 1;
    fhOffset_ = (geomEE->topology().dddConstants()).layers(true);
    wmaxEE    = (geomEE->topology().dddConstants()).waferCount(0);
    auto geomFH = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::HGCalHSi,ForwardSubdetector::ForwardEmpty));
    bhOffset_ = fhOffset_;
    wmaxFH    = (geomFH->topology().dddConstants()).waferCount(0);
  }
  else {
    geometryType_ = 0;
    geomEE = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCEE));
    fhOffset_ = (geomEE->topology().dddConstants()).layers(true);
    wmaxEE    = 1 + (geomEE->topology().dddConstants()).waferMax();
    auto geomFH = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCHEF));
    bhOffset_ = fhOffset_ + (geomFH->topology().dddConstants()).layers(true);
    wmaxFH    = 1 + (geomFH->topology().dddConstants()).waferMax();
  }
  maxNumberOfWafersPerLayer_ = std::max(wmaxEE,wmaxFH);
}

const CaloSubdetectorGeometry* RecHitTools::getSubdetectorGeometry( const DetId& id ) const {
  DetId::Detector det = id.det();
  int subdet = (det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc) ? ForwardSubdetector::ForwardEmpty : id.subdetId();
  auto geom = geom_->getSubdetectorGeometry(det,subdet);
  check_geom(geom);
  return geom;
}

GlobalPoint RecHitTools::getPosition(const DetId& id) const {
  auto geom = getSubdetectorGeometry(id);
  GlobalPoint position;
  if (id.det() == DetId::Hcal) {
    position = geom->getGeometry(id)->getPosition();
  } else {
    auto hg = static_cast<const HGCalGeometry*>(geom);
    position = hg->getPosition(id);
  }
  return position;
}

GlobalPoint RecHitTools::getPositionLayer(unsigned layer) const {
  const int hgcal_default_wafer = 50;
  const int hcal_default_ieta = 50;
  const int hcal_default_iphi = 100;
  DetId id(0);
  if(geometryType_==0){
    if (layer <= fhOffset_) id = HGCalDetId(ForwardSubdetector::HGCEE, 1, layer, 1, hgcal_default_wafer, 1);
    else if (layer <= bhOffset_) id = HGCalDetId(ForwardSubdetector::HGCHEF, 1, layer - fhOffset_, 1, hgcal_default_wafer, 1);
    else  id = HcalDetId(HcalSubdetector::HcalEndcap, hcal_default_ieta, hcal_default_iphi, layer - bhOffset_);
  }
  else {
    if (layer <= fhOffset_) id = HGCSiliconDetId(DetId::HGCalEE, 1, 0, layer, 0, 0, 0, 0);
    else if (layer <= bhOffset_) id = HGCSiliconDetId(DetId::HGCalHSi, 1, 0, layer - fhOffset_, 0, 0, 0, 0);
    else  id = HGCScintillatorDetId(0, layer - bhOffset_, 1, 0);
  }
  return getPosition(id);
}

int RecHitTools::zside(const DetId& id) const {
  int zside = 0;
  if (id.det() == DetId::Forward) {
    zside = HGCalDetId(id).zside();
  } else if( id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
    zside = HcalDetId(id).zside();
  } else if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    zside = HGCSiliconDetId(id).zside();
  } else if (id.det() == DetId::HGCalHSc) {
    zside = HGCScintillatorDetId(id).zside();
  }
  return zside;
}

std::float_t RecHitTools::getSiThickness(const DetId& id) const {
  auto geom = getSubdetectorGeometry(id);
  std::float_t thick(0.37);
  if (id.det() == DetId::Forward) {
    const HGCalDetId hid(id);
    auto ddd = get_ddd(geom,hid);
    thick    = ddd->cellThickness(hid.layer(),hid.wafer(),0);
  } else if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    const HGCSiliconDetId hid(id);
    auto ddd = get_ddd(geom,hid);
    thick    = ddd->cellThickness(hid.layer(),hid.waferU(),hid.waferV());
  } else {
    LogDebug("getSiThickness::InvalidSiliconDetid")
      << "det id: " << std::hex << id.rawId() << std::dec << ":"
      << id.det() << " is not HGCal silicon!";
    // It does not make any sense to return 0.37 as thickness for a detector
    // that is not Silicon(does it?). Mark it as 0. to be able to distinguish
    // the case.
    thick = 0.f;
  }
  return thick;
}

int RecHitTools::getSiThickIndex(const DetId& id) const {
  int thickIndex = -1;
  if (id.det() == DetId::Forward) {
    float thickness = getSiThickness(id);
    if (thickness > 99. && thickness < 101.)
      thickIndex = 0;
    else if (thickness > 199. && thickness < 201.)
      thickIndex = 1;
    else if (thickness > 299. && thickness < 301.)
      thickIndex = 2;
    else
      assert(thickIndex > 0 && "ERROR - silicon thickness has a nonsensical value");
  } else if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    thickIndex = HGCSiliconDetId(id).type();
  }
  return thickIndex;
}

std::float_t RecHitTools::getRadiusToSide(const DetId& id) const {
  auto geom = getSubdetectorGeometry(id);
  std::float_t size(std::numeric_limits<std::float_t>::max());
  if (id.det() == DetId::Forward) {
    const HGCalDetId hid(id);
    auto  ddd = get_ddd(geom,hid);
    size     = ddd->cellSizeHex(hid.waferType());
  } else if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    const HGCSiliconDetId hid(id);
    auto  ddd = get_ddd(geom,hid);
    size     = ddd->cellSizeHex(hid.type());
  } else {
    edm::LogError("getRadiusToSide::InvalidSiliconDetid")
      << "det id: " << std::hex << id.rawId() << std::dec << ":"
      << id.det() << " is not HGCal silicon!";
  }
  return size;
}

unsigned int RecHitTools::getLayer(const ForwardSubdetector type) const {

  int layer;
  switch (type) {
    case(ForwardSubdetector::HGCEE): {
      auto geomEE = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCEE));
      layer       = (geomEE->topology().dddConstants()).layers(true);
      break;
    }
    case (ForwardSubdetector::HGCHEF): {
      auto geomFH = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCHEF));
      layer       = (geomFH->topology().dddConstants()).layers(true);
      break;
    }
    case (ForwardSubdetector::HGCHEB): {
      auto geomBH = static_cast<const HcalGeometry*>(geom_->getSubdetectorGeometry(DetId::Hcal,HcalSubdetector::HcalEndcap));
      layer       = (geomBH->topology().dddConstants())->getMaxDepth(1);
      break;
    }
    case (ForwardSubdetector::ForwardEmpty): {
      auto geomEE = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCEE));
      layer       = (geomEE->topology().dddConstants()).layers(true);
      auto geomFH = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCHEF));
      layer      += (geomFH->topology().dddConstants()).layers(true);
      auto geomBH = static_cast<const HcalGeometry*>(geom_->getSubdetectorGeometry(DetId::Hcal,HcalSubdetector::HcalEndcap));
      layer      += (geomBH->topology().dddConstants())->getMaxDepth(1);
      break;
    }
    default: layer = 0;
  }
  return (unsigned int)(layer);
}

unsigned int RecHitTools::getLayer(const DetId::Detector type) const {

  int layer;
  switch (type) {
  case(DetId::HGCalEE): {
    auto geomEE = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(type,ForwardSubdetector::ForwardEmpty));
    layer       = (geomEE->topology().dddConstants()).layers(true);
    break;
  }
  case (DetId::HGCalHSi): {
    auto geomFH = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(type,ForwardSubdetector::ForwardEmpty));
    layer       = (geomFH->topology().dddConstants()).layers(true);
    break;
  }
  case (DetId::HGCalHSc): {
    auto geomBH = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(type,ForwardSubdetector::ForwardEmpty));
    layer       = (geomBH->topology().dddConstants()).layers(true);
    break;
    }
  case (DetId::Forward): {
    auto geomEE = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::HGCalEE,ForwardSubdetector::ForwardEmpty));
    layer       = (geomEE->topology().dddConstants()).layers(true);
    auto geomFH = static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::HGCalHSi,ForwardSubdetector::ForwardEmpty));
    layer      += (geomFH->topology().dddConstants()).layers(true);
    break;
  }
  default: layer = 0;
  }
  return (unsigned int)(layer);
}

unsigned int RecHitTools::getLayer(const DetId& id) const {
  unsigned int layer = std::numeric_limits<unsigned int>::max();
  if (id.det() == DetId::Forward) {
    layer = HGCalDetId(id).layer();
  } else if (id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
    layer = HcalDetId(id).depth();
  } else if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    layer = HGCSiliconDetId(id).layer();
  } else if (id.det() == DetId::HGCalHSc) {
    layer = HGCScintillatorDetId(id).layer();
  }
  return layer;
}

unsigned int RecHitTools::getLayerWithOffset(const DetId& id) const {
  unsigned int layer = getLayer(id);
  if (id.det() == DetId::Forward && id.subdetId() == HGCHEF ) {
    layer += fhOffset_;
  } else if (id.det() == DetId::HGCalHSi || id.det() == DetId::HGCalHSc) {
    layer += fhOffset_;
  } else if (id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
    layer += bhOffset_;
  }
  return layer;
}

unsigned int RecHitTools::getWafer(const DetId& id) const {
  unsigned int wafer = std::numeric_limits<unsigned int>::max();
  if (id.det() == DetId::Forward) {
    wafer = HGCalDetId(id).wafer();
  }
  else {
    edm::LogError("getWafer::InvalidSiliconDetid")
      << "det id: " << std::hex << id.rawId() << std::dec << ":"
      << id.det() << " is not HGCal silicon!";
  }
  return wafer;
}

unsigned int RecHitTools::getCell(const DetId& id) const {
  unsigned int cell = std::numeric_limits<unsigned int>::max();
  if (id.det() == DetId::Forward) {
    cell = HGCalDetId(id).cell();
  }
  else {
    edm::LogError("getCell::InvalidSiliconDetid")
      << "det id: " << std::hex << id.rawId() << std::dec << ":"
      << id.det() << " is not HGCal silicon!";
  }
  return cell;
}

bool RecHitTools::isHalfCell(const DetId& id) const {
  bool ishalf = false;
  if (id.det() == DetId::Forward) {
    HGCalDetId hid(id);
    auto geom = getSubdetectorGeometry(hid);
    auto ddd = get_ddd(geom,hid);
    const int waferType = ddd->waferTypeT(hid.waferType());
    return ddd->isHalfCell(waferType,hid.cell());
  }
  //new geometry is always false
  return ishalf;
}

float RecHitTools::getEta(const GlobalPoint& position, const float& vertex_z) const {
  GlobalPoint corrected_position = GlobalPoint(position.x(), position.y(), position.z()-vertex_z);
  return corrected_position.eta();
}

float RecHitTools::getEta(const DetId& id, const float& vertex_z) const {
  GlobalPoint position = getPosition(id);
  float eta = getEta(position, vertex_z);
  return eta;
}

float RecHitTools::getPhi(const GlobalPoint& position) const {
  float phi = atan2(position.y(),position.x());
  return phi;
}

float RecHitTools::getPhi(const DetId& id) const {
  GlobalPoint position = getPosition(id);
  float phi = atan2(position.y(),position.x());
  return phi;
}

float RecHitTools::getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z) const {
  float eta = getEta(position, vertex_z);
  float pt = hitEnergy / cosh(eta);
  return pt;
}

float RecHitTools::getPt(const DetId& id, const float& hitEnergy, const float& vertex_z) const {
  GlobalPoint position = getPosition(id);
  float eta = getEta(position, vertex_z);
  float pt = hitEnergy / cosh(eta);
  return pt;
}
