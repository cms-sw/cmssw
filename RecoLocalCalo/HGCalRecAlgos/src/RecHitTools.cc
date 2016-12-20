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

  inline void check_ddd(const HGCalDDDConstants* ddd) {
    if( nullptr == ddd ) {
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

std::float_t RecHitTools::getSiThickness(const DetId& id) const {
  auto ddd = id.subdetId() == HGCEE ? ddd_[0] : ddd_[1];
  check_ddd(ddd);
  int tidx = ddd->waferTypeL(id);
  return idx_to_thickness*tidx;
}

std::float_t RecHitTools::getRadiusToSide(const DetId& id) const {
  auto ddd = id.subdetId() == HGCEE ? ddd_[0] : ddd_[1];
  check_ddd(ddd);
  const HGCalDetId hid(id);
  std::float_t size = ddd->cellSizeHex(hid.waferType());
  return size;
}

bool RecHitTools::isHalfCell(const DetId& id) const {
  auto ddd = id.subdetId() == HGCEE ? ddd_[0] : ddd_[1];
  check_ddd(ddd);
  const HGCalDetId hid(id);
  const int waferType = ddd->waferTypeT(hid.waferType());  
  return ddd->isHalfCell(waferType,hid.cell());
}


