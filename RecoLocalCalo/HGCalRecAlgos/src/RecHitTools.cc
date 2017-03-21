#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
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
  // define layer offsets
  // https://github.com/cms-sw/cmssw/blob/CMSSW_8_1_X/DataFormats/ForwardDetId/interface/ForwardSubdetector.h
  // (EE) HGCEE=3, (FH) HGCHEF=4, BH is HcalEndcap = 2 encoded with HcalDetId
  const unsigned int fhOffset = 28; // number of EE layers
  const unsigned int bhOffset = fhOffset + 12; // number of EE+FH layers

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
  
}

void RecHitTools::getEvent(const edm::Event& ev) {
}

void RecHitTools::getEventSetup(const edm::EventSetup& es) {
  edm::ESHandle<CaloGeometry> geom;
  es.get<CaloGeometryRecord>().get(geom);

  geom_ = geom.product();
}

GlobalPoint RecHitTools::getPosition(const DetId& id) const {
  auto geom = geom_->getSubdetectorGeometry(id);
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

std::float_t RecHitTools::getSiThickness(const DetId& id) const {
  auto geom = geom_->getSubdetectorGeometry(id);
  check_geom(geom);
  if( id.det() != DetId::Forward ) {
    edm::LogError("getSiThickness::InvalidSiliconDetid")
      << "det id: " << id.rawId() << " is not HGCal silicon!";
  }  
  const HGCalDetId hid(id);
  auto ddd = get_ddd(geom,hid);
  unsigned int wafer = hid.wafer();
  int tidx = ddd->waferTypeL(wafer);
  return idx_to_thickness*tidx;
}

std::float_t RecHitTools::getRadiusToSide(const DetId& id) const {
  auto geom = geom_->getSubdetectorGeometry(id);
  check_geom(geom);
  if( id.det() != DetId::Forward ) {
    edm::LogError("getRadiusToSide::InvalidSiliconDetid")
      << "det id: " << id.rawId() << " is not HGCal silicon!";
    return std::numeric_limits<std::float_t>::max();
  }
  const HGCalDetId hid(id);
  auto ddd = get_ddd(geom,hid);
  std::float_t size = ddd->cellSizeHex(hid.waferType());
  return size;
}

unsigned int RecHitTools::getLayer(const DetId& id) const {
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

unsigned int RecHitTools::getLayerWithOffset(const DetId& id) const {
  unsigned int layer = getLayer(id);  
  if( id.det() == DetId::Forward && id.subdetId() == HGCHEF ) {
    layer += fhOffset;
  } else if( id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
    layer += bhOffset;
  }  
  return layer;
}

unsigned int RecHitTools::getWafer(const DetId& id) const {
  if( id.det() != DetId::Forward ) {
    edm::LogError("getWafer::InvalidSiliconDetid")
      << "det id: " << id.rawId() << " is not HGCal silicon!";
    return std::numeric_limits<unsigned int>::max();
  } 
  const HGCalDetId hid(id);
  unsigned int wafer = hid.wafer();
  return wafer;
}

unsigned int RecHitTools::getCell(const DetId& id) const {
  if( id.det() != DetId::Forward ) {
    edm::LogError("getCell::InvalidSiliconDetid")
      << "det id: " << id.rawId() << " is not HGCal silicon!";
    return std::numeric_limits<unsigned int>::max();
  }
  const HGCalDetId hid(id);
  unsigned int cell = hid.cell();
  return cell;
}

bool RecHitTools::isHalfCell(const DetId& id) const {
  if( id.det() != DetId::Forward ) {
    return false;
  }
  auto geom = geom_->getSubdetectorGeometry(id);
  check_geom(geom);  
  const HGCalDetId hid(id);
  auto ddd = get_ddd(geom,hid);
  const int waferType = ddd->waferTypeT(hid.waferType());
  return ddd->isHalfCell(waferType,hid.cell());
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
