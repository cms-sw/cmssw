#include "L1Trigger/L1TMuon/interface/TTGeometryTranslator.h"
#include "L1Trigger/L1TMuon/interface/TTMuonTriggerPrimitive.h"

// event setup stuff / geometries
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

using Phase2TrackerGeomDetUnit = PixelGeomDetUnit;
using Phase2TrackerTopology = PixelTopology;

using namespace L1TMuon;

TTGeometryTranslator::TTGeometryTranslator() : _geom_cache_id(0ULL), _topo_cache_id(0ULL), _magfield_cache_id(0ULL) {}

TTGeometryTranslator::~TTGeometryTranslator() {}

bool TTGeometryTranslator::isBarrel(const TTTriggerPrimitive& tp) const {
  const DetId detId = tp.detId();
  bool isBarrel = (detId.subdetId() == StripSubdetector::TOB);
  //bool isEndcap = (detId.subdetId() == StripSubdetector::TID);
  return isBarrel;
}

bool TTGeometryTranslator::isPSModule(const TTTriggerPrimitive& tp) const {
  const DetId detId = tp.detId();
  const TrackerGeometry::ModuleType moduleType = _geom->getDetectorType(detId);
  bool isPSModule =
      (moduleType == TrackerGeometry::ModuleType::Ph2PSP) || (moduleType == TrackerGeometry::ModuleType::Ph2PSS);
  //bool isSSModule = (moduleType == TrackerGeometry::ModuleType::Ph2SS);
  return isPSModule;
}

int TTGeometryTranslator::region(const TTTriggerPrimitive& tp) const {
  int region = 0;

  const DetId detId = tp.detId();
  if (detId.subdetId() == StripSubdetector::TOB) {  // barrel
    region = 0;
  } else if (detId.subdetId() == StripSubdetector::TID) {  // endcap
    int type = _topo->tidSide(detId);                      // 1=-ve 2=+ve
    if (type == 1) {
      region = -1;
    } else if (type == 2) {
      region = +1;
    }
  }
  return region;
}

int TTGeometryTranslator::layer(const TTTriggerPrimitive& tp) const {
  int layer = 0;

  const DetId detId = tp.detId();
  if (detId.subdetId() == StripSubdetector::TOB) {  // barrel
    layer = static_cast<int>(_topo->layer(detId));
  } else if (detId.subdetId() == StripSubdetector::TID) {  // endcap
    layer = static_cast<int>(_topo->layer(detId));
  }
  return layer;
}

int TTGeometryTranslator::ring(const TTTriggerPrimitive& tp) const {
  int ring = 0;

  const DetId detId = tp.detId();
  if (detId.subdetId() == StripSubdetector::TOB) {  // barrel
    ring = static_cast<int>(_topo->tobRod(detId));
  } else if (detId.subdetId() == StripSubdetector::TID) {  // endcap
    ring = static_cast<int>(_topo->tidRing(detId));
  }
  return ring;
}

int TTGeometryTranslator::module(const TTTriggerPrimitive& tp) const {
  int module = 0;

  const DetId detId = tp.detId();
  if (detId.subdetId() == StripSubdetector::TOB) {  // barrel
    module = static_cast<int>(_topo->module(detId));
  } else if (detId.subdetId() == StripSubdetector::TID) {  // endcap
    module = static_cast<int>(_topo->module(detId));
  }
  return module;
}

double TTGeometryTranslator::calculateGlobalEta(const TTTriggerPrimitive& tp) const {
  switch (tp.subsystem()) {
    case TTTriggerPrimitive::kTT:
      return calcTTSpecificEta(tp);
      break;
    default:
      return std::nan("Invalid TP type!");
      break;
  }
}

double TTGeometryTranslator::calculateGlobalPhi(const TTTriggerPrimitive& tp) const {
  switch (tp.subsystem()) {
    case TTTriggerPrimitive::kTT:
      return calcTTSpecificPhi(tp);
      break;
    default:
      return std::nan("Invalid TP type!");
      break;
  }
}

double TTGeometryTranslator::calculateBendAngle(const TTTriggerPrimitive& tp) const {
  switch (tp.subsystem()) {
    case TTTriggerPrimitive::kTT:
      return calcTTSpecificBend(tp);
      break;
    default:
      return std::nan("Invalid TP type!");
      break;
  }
}

GlobalPoint TTGeometryTranslator::getGlobalPoint(const TTTriggerPrimitive& tp) const {
  switch (tp.subsystem()) {
    case TTTriggerPrimitive::kTT:
      return getTTSpecificPoint(tp);
      break;
    default:
      GlobalPoint ret(
          GlobalPoint::Polar(std::nan("Invalid TP type!"), std::nan("Invalid TP type!"), std::nan("Invalid TP type!")));
      return ret;
      break;
  }
}

void TTGeometryTranslator::checkAndUpdateGeometry(const edm::EventSetup& es) {
  const TrackerDigiGeometryRecord& geom = es.get<TrackerDigiGeometryRecord>();
  unsigned long long geomid = geom.cacheIdentifier();
  if (_geom_cache_id != geomid) {
    geom.get(_geom);
    _geom_cache_id = geomid;
  }

  const TrackerTopologyRcd& topo = es.get<TrackerTopologyRcd>();
  unsigned long long topoid = topo.cacheIdentifier();
  if (_topo_cache_id != topoid) {
    topo.get(_topo);
    _topo_cache_id = topoid;
  }

  const IdealMagneticFieldRecord& magfield = es.get<IdealMagneticFieldRecord>();
  unsigned long long magfieldid = magfield.cacheIdentifier();
  if (_magfield_cache_id != magfieldid) {
    magfield.get(_magfield);
    _magfield_cache_id = magfieldid;
  }
}

GlobalPoint TTGeometryTranslator::getTTSpecificPoint(const TTTriggerPrimitive& tp) const {
  // Check L1Trigger/TrackTrigger/src/TTStubAlgorithm_official.cc
  const DetId detId = tp.detId();
  const GeomDetUnit* geoUnit = _geom->idToDetUnit(detId + 1);  // det0
  //const GeomDetUnit* geoUnit = _geom->idToDetUnit(detId+2);  // det1
  const Phase2TrackerGeomDetUnit* ph2TkGeoUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(geoUnit);
  const MeasurementPoint mp(tp.getTTData().row_f, tp.getTTData().col_f);
  const GlobalPoint gp = ph2TkGeoUnit->surface().toGlobal(ph2TkGeoUnit->specificTopology().localPosition(mp));
  return gp;
}

double TTGeometryTranslator::calcTTSpecificEta(const TTTriggerPrimitive& tp) const {
  return getTTSpecificPoint(tp).eta();
}

double TTGeometryTranslator::calcTTSpecificPhi(const TTTriggerPrimitive& tp) const {
  return getTTSpecificPoint(tp).phi();
}

double TTGeometryTranslator::calcTTSpecificBend(const TTTriggerPrimitive& tp) const { return tp.getTTData().bend; }
