#include "RecoLocalFastTime/FTLCommonAlgos/interface/RecHitTools.h"

#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"

#include "Geometry/Records/interface/FastTimeGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace ftl;

void RecHitTools::getEvent(const edm::Event& ev) {
}

void RecHitTools::getEventSetup(const edm::EventSetup& es) {
  edm::ESHandle<FastTimeGeometry> ftgeom;
  es.get<FastTimeGeometryRecord>().get(ftgeom);
  geom_ = ftgeom.product();
  ddd_  = &(geom_->topology().dddConstants()); 
}

const GlobalPoint& RecHitTools::getPosition(const DetId& id) const {
  return geom_->getGeometry(id)->getPosition();
}

const FlatTrd::CornersVec& RecHitTools::getCorners(const DetId& id) const {
  return geom_->getGeometry(id)->getCorners();
}

RecHitTools::HitType RecHitTools::getHitType(const DetId& id) const {
  FastTimeDetId fid(id);
  return (HitType)fid.type();
}





