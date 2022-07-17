#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HFNoseTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerModuleDetId.h"
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

// Kept for backward compatibility: used in L1Trigger/L1CaloTrigger/test
void HGCalTriggerTools::eventSetup(const edm::EventSetup& es,
                                   const edm::ESGetToken<HGCalTriggerGeometryBase, CaloGeometryRecord>& token) {
  const edm::ESHandle<HGCalTriggerGeometryBase>& triggerGeometry = es.getHandle(token);
  setGeometry(triggerGeometry.product());
}

void HGCalTriggerTools::setGeometry(const HGCalTriggerGeometryBase* const geom) {
  geom_ = geom;
  eeLayers_ = geom_->eeTopology().dddConstants().layers(true);
  fhLayers_ = geom_->fhTopology().dddConstants().layers(true);
  if (geom_->isWithNoseGeometry())
    noseLayers_ = geom_->noseTopology().dddConstants().layers(true);

  bhLayers_ = geom_->hscTopology().dddConstants().layers(true);
  totalLayers_ = eeLayers_ + fhLayers_;
}

GlobalPoint HGCalTriggerTools::getTCPosition(const DetId& id) const {
  if (id.det() == DetId::HGCalEE) {
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
  if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose) {
    layer = HFNoseDetId(id).layer();
  } else if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HGCTrigger) {
    layer = HGCalTriggerModuleDetId(id).layer();
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
  } else if (isHad(id)) {
    l += eeLayers_;
  }

  return l;
}

bool HGCalTriggerTools::isEm(const DetId& id) const {
  bool em = false;

  if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose) {
    em = HFNoseDetId(id).isEE();
  } else if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HGCTrigger) {
    em = HGCalTriggerModuleDetId(id).isEE();
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
  if (id.det() == DetId::Forward && id.subdetId() == HGCTrigger) {
    silicon = (HGCalTriggerModuleDetId(id).triggerSubdetId() != HGCalHScTrigger);
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

HGCalTriggerTools::SubDetectorType HGCalTriggerTools::getSubDetectorType(const DetId& id) const {
  SubDetectorType subdet;
  if (!isScintillator(id)) {
    if (isEm(id))
      subdet = HGCalTriggerTools::hgcal_silicon_CEE;
    else
      subdet = HGCalTriggerTools::hgcal_silicon_CEH;
  } else
    subdet = HGCalTriggerTools::hgcal_scintillator;
  return subdet;
}

int HGCalTriggerTools::zside(const DetId& id) const {
  int zside = 0;
  if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose) {
    zside = HFNoseDetId(id).zside();
  } else if (id.det() == DetId::Forward && id.subdetId() == ForwardSubdetector::HGCTrigger) {
    zside = HGCalTriggerModuleDetId(id).zside();
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

int HGCalTriggerTools::thicknessIndex(const DetId& id) const {
  if (isScintillator(id)) {
    return kScintillatorPseudoThicknessIndex_;
  }
  unsigned det = id.det();
  int thickness = 0;
  if (det == DetId::HGCalEE || det == DetId::HGCalHSi) {
    thickness = HGCSiliconDetId(id).type();
  } else if (det == DetId::Forward && id.subdetId() == ForwardSubdetector::HFNose) {
    thickness = HFNoseDetId(id).type();
  } else if (det == DetId::Forward && id.subdetId() == ForwardSubdetector::HGCTrigger) {
    thickness = HGCalTriggerModuleDetId(id).type();
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
  } else if ((subdet == ForwardSubdetector::HGCHEB) || (subdet == DetId::HGCalHSc)) {
    layerGlobalZ = geom_->hscTopology().dddConstants().waferZ(layer, true);
  }
  return layerGlobalZ;
}

DetId HGCalTriggerTools::simToReco(const DetId& simid, const HGCalTopology& topo) const {
  DetId recoid(0);
  const auto& dddConst = topo.dddConstants();
  if (dddConst.waferHexagon8() || dddConst.tileTrapezoid()) {
    recoid = simid;
  }
  return recoid;
}
