#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
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
  // HGCEE=3, HGCHEF=4, HGCHEB=5
  const unsigned int hefOffset = 28; // number of EE layers
  const unsigned int hebOffset = hefOffset + 12; // number of EE+FH layers


  inline void check_ddd(const HGCalDDDConstants* ddd) {
    if( nullptr == ddd ) {
      throw cms::Exception("hgcal::RecHitTools")
        << "HGCalGeometry not provided yet to hgcal::RecHitTools!";
    }
  }

  inline void check_geom(const HGCalGeometry* geom) {
    if( nullptr == geom ) {
      throw cms::Exception("hgcal::RecHitTools")
        << "HGCalGeometry not provided yet to hgcal::RecHitTools!";
    }
  }
}

void RecHitTools::getEvent(const edm::Event& ev) {
}

void RecHitTools::getEventSetup(const edm::EventSetup& es) {
  edm::ESHandle<HGCalGeometry> hgeom;
  es.get<IdealGeometryRecord>().get(hgcalee_sens,hgeom);
  geom_[0] = hgeom.product();
  ddd_[0]  = &(geom_[0]->topology().dddConstants());
  es.get<IdealGeometryRecord>().get(hgcalfh_sens,hgeom);
  geom_[1] = hgeom.product();
  ddd_[1]  = &(geom_[0]->topology().dddConstants());
}

GlobalPoint RecHitTools::getPosition(const DetId& id) const {
  auto geom = id.subdetId() == HGCEE ? geom_[0] : geom_[1];
  check_geom(geom);
  GlobalPoint position( std::move( geom->getPosition( id ) ) );
  return position;
}

std::float_t RecHitTools::getSiThickness(const DetId& id) const {
  auto ddd = id.subdetId() == HGCEE ? ddd_[0] : ddd_[1];
  check_ddd(ddd);
  const HGCalDetId hid(id);
  unsigned int wafer = hid.wafer();
  int tidx = ddd->waferTypeL(wafer);
  return idx_to_thickness*tidx;
}

std::float_t RecHitTools::getRadiusToSide(const DetId& id) const {
  auto ddd = id.subdetId() == HGCEE ? ddd_[0] : ddd_[1];
  check_ddd(ddd);
  const HGCalDetId hid(id);
  std::float_t size = ddd->cellSizeHex(hid.waferType());
  return size;
}

unsigned int RecHitTools::getLayer(const DetId& id) const {
  auto ddd = id.subdetId() == HGCEE ? ddd_[0] : ddd_[1];
  check_ddd(ddd);
  const HGCalDetId hid(id);
  unsigned int layer = hid.layer();
  return layer;
}

unsigned int RecHitTools::getLayerWithOffset(const DetId& id) const {
  auto ddd = id.subdetId() == HGCEE ? ddd_[0] : ddd_[1];
  check_ddd(ddd);
  const HGCalDetId hid(id);
  unsigned int layer = hid.layer();
  unsigned int offset = 0;
  switch(id.subdetId()) {
      case HGCHEF:
        offset += hefOffset;
        break;
      case HGCHEB:
        offset += hebOffset;
        break;
  }
  layer += offset;
  return layer;
}

unsigned int RecHitTools::getWafer(const DetId& id) const {
  auto ddd = id.subdetId() == HGCEE ? ddd_[0] : ddd_[1];
  check_ddd(ddd);
  const HGCalDetId hid(id);
  unsigned int wafer = hid.wafer();
  return wafer;
}

unsigned int RecHitTools::getCell(const DetId& id) const {
  auto ddd = id.subdetId() == HGCEE ? ddd_[0] : ddd_[1];
  check_ddd(ddd);
  const HGCalDetId hid(id);
  unsigned int cell = hid.cell();
  return cell;
}

bool RecHitTools::isHalfCell(const DetId& id) const {
  auto ddd = id.subdetId() == HGCEE ? ddd_[0] : ddd_[1];
  check_ddd(ddd);
  const HGCalDetId hid(id);
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
