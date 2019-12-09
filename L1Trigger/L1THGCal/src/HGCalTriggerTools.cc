#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HFNoseTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/EventSetup.h"

namespace {
  template <typename DDD>
  inline void check_ddd(const DDD* ddd) {
    if (nullptr == ddd) {
      throw cms::Exception("hgcal::HGCalTriggerTools") << "DDDConstants not accessible to hgcal::HGCalTriggerTools!";
    }
  }

  template <typename GEOM>
  inline void check_geom(const GEOM* geom) {
    if (nullptr == geom) {
      throw cms::Exception("hgcal::HGCalTriggerTools") << "Geometry not provided yet to hgcal::HGCalTriggerTools!";
    }
  }
}  // namespace

void HGCalTriggerTools::eventSetup(const edm::EventSetup& es) {
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  es.get<CaloGeometryRecord>().get(triggerGeometry_);
  geom_ = triggerGeometry_.product();

  eeLayers_ = geom_->eeTopology().dddConstants().layers(true);
  fhLayers_ = geom_->fhTopology().dddConstants().layers(true);
  if (geom_->isWithNoseGeometry())
    noseLayers_ = geom_->noseTopology().dddConstants().layers(true);

  if (geom_->isV9Geometry()) {
    bhLayers_ = geom_->hscTopology().dddConstants().layers(true);
    totalLayers_ = eeLayers_ + fhLayers_;
  } else {
    bhLayers_ = geom_->bhTopology().dddConstants()->getMaxDepth(1);
    totalLayers_ = eeLayers_ + fhLayers_ + bhLayers_;
  }
}

GlobalPoint HGCalTriggerTools::getTCPosition(const DetId& id) const {
  if (id.det() == DetId::Hcal || id.det() == DetId::HGCalEE) {
    throw cms::Exception("hgcal::HGCalTriggerTools") << "method getTCPosition called for DetId not belonging to a TC";
    // FIXME: this would actually need a better test...but at the moment I can not think to anything better
    // to distinguish a TC detId
  }

  GlobalPoint position = geom_->getTriggerCellPosition(id);
  return position;
}

unsigned HGCalTriggerTools::layers(ForwardSubdetector type) const {
  unsigned layers = 0;
  switch (type) {
    case ForwardSubdetector::HGCEE:
      layers = eeLayers_;
      break;
    case ForwardSubdetector::HGCHEF:
      layers = fhLayers_;
      break;
    case ForwardSubdetector::HGCHEB:
      layers = bhLayers_;
      break;
    case ForwardSubdetector::HFNose:
      layers = noseLayers_;
      break;
    case ForwardSubdetector::ForwardEmpty:
      layers = totalLayers_;
      break;
    default:
      break;
  };
  return layers;
}

unsigned HGCalTriggerTools::layers(DetId::Detector type) const {
  unsigned layers = 0;
  switch (type) {
    case DetId::HGCalEE:
      layers = eeLayers_;
      break;
    case DetId::HGCalHSi:
      layers = fhLayers_;
      break;
    case DetId::HGCalHSc:
      layers = bhLayers_;
      break;
    // FIXME: to do HFNose
    case DetId::Forward:
      layers = totalLayers_;
      break;
    default:
      break;
  }
  return layers;
}

unsigned HGCalTriggerTools::layer(const DetId& id) const {
  unsigned int layer = std::numeric_limits<unsigned int>::max();
  if (id.det() == DetId::Forward && id.subdetId() != ForwardSubdetector::HFNose) {
    layer = HGCalDetId(id).layer();
  } else if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose) {
    layer = HFNoseDetId(id).layer();
  } else if (id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
    layer = HcalDetId(id).depth();
  } else if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    layer = HGCSiliconDetId(id).layer();
  } else if (id.det() == DetId::HGCalTrigger &&
             (HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalEETrigger ||
              HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalHSiTrigger)) {
    layer = HGCalTriggerDetId(id).layer();
  } else if (id.det() == DetId::HGCalTrigger &&
             HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HFNoseTrigger) {
    layer = HFNoseTriggerDetId(id).layer();
  } else if (id.det() == DetId::HGCalHSc) {
    layer = HGCScintillatorDetId(id).layer();
  }
  return layer;
}

unsigned HGCalTriggerTools::layerWithOffset(const DetId& id) const {
  unsigned int l = layer(id);

  if (isNose(id)) {
    l = layer(id);  // no offset for HFnose
  } else if (isHad(id) && isSilicon(id)) {
    l += eeLayers_;
  } else if (isHad(id) && isScintillator(id)) {
    if (geom_->isV9Geometry())
      l += eeLayers_;  // mixed silicon and scintillator layers
    else
      l += eeLayers_ + fhLayers_;
  }

  return l;
}

bool HGCalTriggerTools::isEm(const DetId& id) const {
  bool em = false;

  if (id.det() == DetId::Forward && id.subdetId() != ForwardSubdetector::HFNose) {
    em = (id.subdetId() == HGCEE);
  } else if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose) {
    em = HFNoseDetId(id).isEE();
  } else if (id.det() == DetId::HGCalEE) {
    em = true;
  } else if (id.det() == DetId::HGCalTrigger &&
             HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalEETrigger) {
    em = true;
  } else if (id.det() == DetId::HGCalTrigger &&
             HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HFNoseTrigger) {
    em = HFNoseTriggerDetId(id).isEE();
  }
  return em;
}

bool HGCalTriggerTools::isNose(const DetId& id) const {
  bool nose = false;
  if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose) {
    nose = true;
  } else if (id.det() == DetId::HGCalTrigger &&
             HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HFNoseTrigger) {
    nose = true;
  }
  return nose;
}

bool HGCalTriggerTools::isSilicon(const DetId& id) const {
  bool silicon = false;
  if (id.det() == DetId::Forward) {
    silicon = (id.subdetId() != HGCHEB);
  } else if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    silicon = true;
  } else if (id.det() == DetId::HGCalTrigger &&
             HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HFNoseTrigger) {
    silicon = true;
  } else if (id.det() == DetId::HGCalTrigger &&
             (HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalEETrigger ||
              HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalHSiTrigger)) {
    silicon = true;
  }
  return silicon;
}

int HGCalTriggerTools::zside(const DetId& id) const {
  int zside = 0;
  if (id.det() == DetId::Forward && id.subdetId() != ForwardSubdetector::HFNose) {
    zside = HGCalDetId(id).zside();
  } else if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose) {
    zside = HFNoseDetId(id).zside();
  } else if (id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
    zside = HcalDetId(id).zside();
  } else if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    zside = HGCSiliconDetId(id).zside();
  } else if (id.det() == DetId::HGCalTrigger &&
             (HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalEETrigger ||
              HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalHSiTrigger)) {
    zside = HGCalTriggerDetId(id).zside();
  } else if (id.det() == DetId::HGCalTrigger &&
             HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HFNoseTrigger) {
    zside = HFNoseTriggerDetId(id).zside();
  } else if (id.det() == DetId::HGCalHSc) {
    zside = HGCScintillatorDetId(id).zside();
  }
  return zside;
}

int HGCalTriggerTools::thicknessIndex(const DetId& id, bool tc) const {
  if (isScintillator(id)) {
    return kScintillatorPseudoThicknessIndex_;
  }
  unsigned det = id.det();
  int thickness = 0;
  // For the v8 detid scheme
  if (det == DetId::Forward && id.subdetId() != ForwardSubdetector::HFNose) {
    if (!tc)
      thickness = sensorCellThicknessV8(id);
    else {
      // For the old geometry, TCs can contain sensor cells
      // with different thicknesses.
      // Use a majority logic to find the TC thickness
      std::array<unsigned, 3> occurences = {{0, 0, 0}};
      for (const auto& c_id : geom_->getCellsFromTriggerCell(id)) {
        unsigned c_det = DetId(c_id).det();
        int c_thickness = -1;
        // For the v8 detid scheme
        if (c_det == DetId::Forward) {
          c_thickness = sensorCellThicknessV8(c_id);
        } else {
          c_thickness = HGCSiliconDetId(c_id).type();
        }
        if (c_thickness < 0 || unsigned(c_thickness) >= occurences.size()) {
          throw cms::Exception("OutOfBound") << "Found thickness index = " << c_thickness;
        }
        occurences[c_thickness]++;
      }
      thickness = std::max_element(occurences.begin(), occurences.end()) - occurences.begin();
    }
  }
  // For the v9 detid scheme
  else if (det == DetId::HGCalEE || det == DetId::HGCalHSi) {
    thickness = HGCSiliconDetId(id).type();
  } else if (det == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose) {
    thickness = HFNoseDetId(id).type();
  } else if (id.det() == DetId::HGCalTrigger &&
             (HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalEETrigger ||
              HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HGCalHSiTrigger)) {
    thickness = HGCalTriggerDetId(id).type();
  } else if (id.det() == DetId::HGCalTrigger &&
             HGCalTriggerDetId(id).subdet() == HGCalTriggerSubdetector::HFNoseTrigger) {
    thickness = HFNoseTriggerDetId(id).type();
  }
  return thickness;
}

float HGCalTriggerTools::getEta(const GlobalPoint& position, const float& vertex_z) const {
  GlobalPoint corrected_position = GlobalPoint(position.x(), position.y(), position.z() - vertex_z);
  return corrected_position.eta();
}

float HGCalTriggerTools::getTCEta(const DetId& id, const float& vertex_z) const {
  GlobalPoint position = getTCPosition(id);
  return getEta(position, vertex_z);
}

float HGCalTriggerTools::getPhi(const GlobalPoint& position) const {
  float phi = atan2(position.y(), position.x());
  return phi;
}

float HGCalTriggerTools::getTCPhi(const DetId& id) const {
  GlobalPoint position = getTCPosition(id);
  return getPhi(position);
}

float HGCalTriggerTools::getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z) const {
  float eta = getEta(position, vertex_z);
  float pt = hitEnergy / cosh(eta);
  return pt;
}

float HGCalTriggerTools::getTCPt(const DetId& id, const float& hitEnergy, const float& vertex_z) const {
  GlobalPoint position = getTCPosition(id);
  return getPt(position, hitEnergy, vertex_z);
}

float HGCalTriggerTools::getLayerZ(const unsigned& layerWithOffset) const {
  int subdet = ForwardSubdetector::HGCEE;
  unsigned offset = 0;
  if (layerWithOffset > lastLayerEE() && layerWithOffset <= lastLayerFH()) {
    subdet = ForwardSubdetector::HGCHEF;
    offset = lastLayerEE();
  } else if (layerWithOffset > lastLayerFH()) {
    subdet = HcalSubdetector::HcalEndcap;
    offset = lastLayerFH();
  }
  // note for HFnose offset is always zero since we have less layers than HGCEE
  return getLayerZ(subdet, layerWithOffset - offset);
}

float HGCalTriggerTools::getLayerZ(const int& subdet, const unsigned& layer) const {
  float layerGlobalZ = 0.;
  if ((subdet == ForwardSubdetector::HGCEE) || (subdet == DetId::HGCalEE)) {
    layerGlobalZ = geom_->eeTopology().dddConstants().waferZ(layer, true);
  } else if ((subdet == ForwardSubdetector::HGCHEF) || (subdet == DetId::HGCalHSi)) {
    layerGlobalZ = geom_->fhTopology().dddConstants().waferZ(layer, true);
  } else if (subdet == ForwardSubdetector::HFNose) {
    layerGlobalZ = geom_->noseTopology().dddConstants().waferZ(layer, true);
  } else if ((subdet == HcalSubdetector::HcalEndcap) || (subdet == ForwardSubdetector::HGCHEB) ||
             (subdet == DetId::HGCalHSc)) {
    if (geom_->isV9Geometry()) {
      layerGlobalZ = geom_->hscTopology().dddConstants().waferZ(layer, true);
    } else {
      layerGlobalZ = geom_->bhTopology().dddConstants()->getRZ(
          HcalSubdetector::HcalEndcap, geom_->bhTopology().dddConstants()->getEtaRange(1).second, layer);
    }
  }
  return layerGlobalZ;
}

DetId HGCalTriggerTools::simToReco(const DetId& simid, const HGCalTopology& topo) const {
  DetId recoid(0);
  const auto& dddConst = topo.dddConstants();
  // V9
  if (dddConst.geomMode() == HGCalGeometryMode::Hexagon8 || dddConst.geomMode() == HGCalGeometryMode::Hexagon8Full ||
      dddConst.geomMode() == HGCalGeometryMode::Trapezoid) {
    recoid = simid;
  }
  // V8
  else {
    int subdet(simid.subdetId());
    int layer = 0, cell = 0, sec = 0, subsec = 0, zp = 0;
    HGCalTestNumbering::unpackHexagonIndex(simid, subdet, zp, layer, sec, subsec, cell);
    // sec is wafer and subsec is celltype
    // skip this hit if after ganging it is not valid
    auto recoLayerCell = dddConst.simToReco(cell, layer, sec, topo.detectorType());
    cell = recoLayerCell.first;
    layer = recoLayerCell.second;
    if (layer >= 0 && cell >= 0) {
      recoid = HGCalDetId((ForwardSubdetector)subdet, zp, layer, subsec, sec, cell);
    }
  }
  return recoid;
}

DetId HGCalTriggerTools::simToReco(const DetId& simid, const HcalTopology& topo) const {
  DetId recoid(0);
  const auto& dddConst = topo.dddConstants();
  HcalDetId id = HcalHitRelabeller::relabel(simid, dddConst);
  if (id.subdet() == int(HcalEndcap))
    recoid = id;
  return recoid;
}

int HGCalTriggerTools::sensorCellThicknessV8(const DetId& id) const {
  int thickness = 0;
  switch (id.subdetId()) {
    case ForwardSubdetector::HGCEE:
      thickness = geom_->eeTopology().dddConstants().waferTypeL(HGCalDetId(id).wafer()) - 1;
      break;
    case ForwardSubdetector::HGCHEF:
      thickness = geom_->fhTopology().dddConstants().waferTypeL(HGCalDetId(id).wafer()) - 1;
      break;
    default:
      break;
  };
  return thickness;
}
