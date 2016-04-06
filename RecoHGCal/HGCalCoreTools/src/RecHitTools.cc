#include "RecoHGCal/HGCalCoreTools/interface/RecHitTools.h"

#include "DataFormats/ForwardDetId/inteface/HGCalDetId.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace hgcal;

namespace {
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
}

std::float_t RecHitTools::getSiThickness(const DetId& id) const {
  check_ddd(ddd_);
  int tidx = ddd_->getWaferTypeL(id);
  return idx_to_thickness*tidx;
}

std::float_t RecHitTools::getRadiusToSide(const DetId& id) const {
  check_ddd(ddd_);
  HGCalDetId hid(id);
  std::float_t size = ddd_->cellSizeHex(hid.waferType());
  return size;
}

bool RecHitTools::isHalfCell(const DetId& id) const {
  check_ddd(ddd_);
  return false;
}


