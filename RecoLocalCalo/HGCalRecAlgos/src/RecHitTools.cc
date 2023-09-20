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
  template <typename DDD>
  inline void check_ddd(const DDD* ddd) {
    if (nullptr == ddd) {
      throw cms::Exception("hgcal::RecHitTools") << "DDDConstants not accessibl to hgcal::RecHitTools!";
    }
  }

  template <typename GEOM>
  inline void check_geom(const GEOM* geom) {
    if (nullptr == geom) {
      throw cms::Exception("hgcal::RecHitTools") << "Geometry not provided yet to hgcal::RecHitTools!";
    }
  }

  inline const HGCalDDDConstants* get_ddd(const CaloSubdetectorGeometry* geom, const HGCalDetId& detid) {
    const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom);
    const HGCalDDDConstants* ddd = &(hg->topology().dddConstants());
    check_ddd(ddd);
    return ddd;
  }

  inline const HGCalDDDConstants* get_ddd(const CaloSubdetectorGeometry* geom, const HGCSiliconDetId& detid) {
    const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom);
    const HGCalDDDConstants* ddd = &(hg->topology().dddConstants());
    check_ddd(ddd);
    return ddd;
  }

  inline const HGCalDDDConstants* get_ddd(const CaloSubdetectorGeometry* geom, const HFNoseDetId& detid) {
    const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom);
    const HGCalDDDConstants* ddd = &(hg->topology().dddConstants());
    check_ddd(ddd);
    return ddd;
  }

  inline const HGCalDDDConstants* get_ddd(const CaloGeometry* geom, int type, int maxLayerEE, int layer) {
    DetId::Detector det = ((type == 0) ? DetId::Forward : ((layer > maxLayerEE) ? DetId::HGCalHSi : DetId::HGCalEE));
    int subdet = ((type != 0) ? ForwardSubdetector::ForwardEmpty
                              : ((layer > maxLayerEE) ? ForwardSubdetector::HGCEE : ForwardSubdetector::HGCHEF));
    const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(det, subdet));
    const HGCalDDDConstants* ddd = &(hg->topology().dddConstants());
    check_ddd(ddd);
    return ddd;
  }

  enum LayerType {
    CE_E_120 = 0,
    CE_E_200 = 1,
    CE_E_300 = 2,
    CE_H_120 = 3,
    CE_H_200 = 4,
    CE_H_300 = 5,
    CE_H_SCINT = 6,
    EnumSize = 7
  };

}  // namespace

void RecHitTools::setGeometry(const CaloGeometry& geom) {
  geom_ = &geom;
  unsigned int wmaxEE(0), wmaxFH(0);
  auto geomEE = static_cast<const HGCalGeometry*>(
      geom_->getSubdetectorGeometry(DetId::HGCalEE, ForwardSubdetector::ForwardEmpty));
  //check if it's the new geometry
  if (geomEE) {
    geometryType_ = 1;
    eeOffset_ = (geomEE->topology().dddConstants()).getLayerOffset();
    wmaxEE = (geomEE->topology().dddConstants()).waferCount(0);
    auto geomFH = static_cast<const HGCalGeometry*>(
        geom_->getSubdetectorGeometry(DetId::HGCalHSi, ForwardSubdetector::ForwardEmpty));
    fhOffset_ = (geomFH->topology().dddConstants()).getLayerOffset();
    wmaxFH = (geomFH->topology().dddConstants()).waferCount(0);
    fhLastLayer_ = fhOffset_ + (geomFH->topology().dddConstants()).lastLayer(true);
    auto geomBH = static_cast<const HGCalGeometry*>(
        geom_->getSubdetectorGeometry(DetId::HGCalHSc, ForwardSubdetector::ForwardEmpty));
    bhOffset_ = (geomBH->topology().dddConstants()).getLayerOffset();
    bhFirstLayer_ = bhOffset_ + (geomBH->topology().dddConstants()).firstLayer();
    bhLastLayer_ = bhOffset_ + (geomBH->topology().dddConstants()).lastLayer(true);
    bhMaxIphi_ = geomBH->topology().dddConstants().maxCells(true);
  } else {
    geometryType_ = 0;
    geomEE =
        static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HGCEE));
    eeOffset_ = (geomEE->topology().dddConstants()).getLayerOffset();
    wmaxEE = 1 + (geomEE->topology().dddConstants()).waferMax();
    auto geomFH =
        static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HGCHEF));
    fhOffset_ = (geomFH->topology().dddConstants()).getLayerOffset();
    fhLastLayer_ = fhOffset_ + (geomFH->topology().dddConstants()).layers(true);
    bhOffset_ = fhLastLayer_;
    bhFirstLayer_ = bhOffset_ + 1;
    wmaxFH = 1 + (geomFH->topology().dddConstants()).waferMax();
    auto geomBH =
        static_cast<const HcalGeometry*>(geom_->getSubdetectorGeometry(DetId::Hcal, HcalSubdetector::HcalEndcap));
    bhLastLayer_ = bhOffset_ + (geomBH->topology().dddConstants())->getMaxDepth(1);
  }
  maxNumberOfWafersPerLayer_ = std::max(wmaxEE, wmaxFH);
  // For nose geometry
  auto geomNose =
      static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HFNose));
  if (geomNose) {
    maxNumberOfWafersNose_ = (geomNose->topology().dddConstants()).waferCount(0);
    noseLastLayer_ = (geomNose->topology().dddConstants()).layers(true);
  } else {
    maxNumberOfWafersNose_ = 0;
    noseLastLayer_ = 0;
  }
}

const CaloSubdetectorGeometry* RecHitTools::getSubdetectorGeometry(const DetId& id) const {
  DetId::Detector det = id.det();
  int subdet = (det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc)
                   ? ForwardSubdetector::ForwardEmpty
                   : id.subdetId();
  auto geom = geom_->getSubdetectorGeometry(det, subdet);
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

GlobalPoint RecHitTools::getPositionLayer(int layer, bool nose) const {
  unsigned int lay = std::abs(layer);
  double z(0);
  if (nose) {
    auto geomNose =
        static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HFNose));
    if (geomNose) {
      const HGCalDDDConstants* ddd = &(geomNose->topology().dddConstants());
      if (ddd)
        z = (layer > 0) ? ddd->waferZ(lay, true) : -ddd->waferZ(lay, true);
    }
  } else {
    const HGCalDDDConstants* ddd = get_ddd(geom_, geometryType_, fhOffset_, lay);
    if (geometryType_ == 1) {
      if (lay > fhOffset_)
        lay -= fhOffset_;
    }
    z = (layer > 0) ? ddd->waferZ(lay, true) : -ddd->waferZ(lay, true);
  }
  return GlobalPoint(0, 0, z);
}

int RecHitTools::zside(const DetId& id) const {
  int zside = 0;
  if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    zside = HGCSiliconDetId(id).zside();
  } else if (id.det() == DetId::HGCalHSc) {
    zside = HGCScintillatorDetId(id).zside();
  } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(HFNose)) {
    zside = HFNoseDetId(id).zside();
  } else if (id.det() == DetId::Forward) {
    zside = HGCalDetId(id).zside();
  } else if (id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
    zside = HcalDetId(id).zside();
  }
  return zside;
}

std::float_t RecHitTools::getSiThickness(const DetId& id) const {
  auto geom = getSubdetectorGeometry(id);
  std::float_t thick(0.37);
  if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    const HGCSiliconDetId hid(id);
    auto ddd = get_ddd(geom, hid);
    thick = ddd->cellThickness(hid.layer(), hid.waferU(), hid.waferV());
  } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(HFNose)) {
    const HFNoseDetId hid(id);
    auto ddd = get_ddd(geom, hid);
    thick = ddd->cellThickness(hid.layer(), hid.waferU(), hid.waferV());
  } else if (id.det() == DetId::Forward) {
    const HGCalDetId hid(id);
    auto ddd = get_ddd(geom, hid);
    thick = ddd->cellThickness(hid.layer(), hid.wafer(), 0);
  } else {
    LogDebug("getSiThickness::InvalidSiliconDetid")
        << "det id: " << std::hex << id.rawId() << std::dec << ":" << id.det() << " is not HGCal silicon!";
    // It does not make any sense to return 0.37 as thickness for a detector
    // that is not Silicon(does it?). Mark it as 0. to be able to distinguish
    // the case.
    thick = 0.f;
  }
  return thick;
}

int RecHitTools::getSiThickIndex(const DetId& id) const {
  int thickIndex = -1;
  if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    thickIndex = HGCSiliconDetId(id).type();
  } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(HFNose)) {
    thickIndex = HFNoseDetId(id).type();
  } else if (id.det() == DetId::Forward) {
    float thickness = getSiThickness(id);
    if (thickness > 99. && thickness < 121.)
      thickIndex = 0;
    else if (thickness > 199. && thickness < 201.)
      thickIndex = 1;
    else if (thickness > 299. && thickness < 301.)
      thickIndex = 2;
    else
      assert(thickIndex > 0 && "ERROR - silicon thickness has a nonsensical value");
  }
  return thickIndex;
}

std::pair<float, float> RecHitTools::getScintDEtaDPhi(const DetId& id) const {
  if (!isScintillator(id)) {
    LogDebug("getScintDEtaDPhi::InvalidScintDetid")
        << "det id: " << std::hex << id.rawId() << std::dec << ":" << id.det() << " is not HGCal scintillator!";
    return {0.f, 0.f};
  }
  auto cellGeom = getSubdetectorGeometry(id)->getGeometry(id);
  return {cellGeom->etaSpan(), cellGeom->phiSpan()};
}

std::float_t RecHitTools::getRadiusToSide(const DetId& id) const {
  auto geom = getSubdetectorGeometry(id);
  std::float_t size(std::numeric_limits<std::float_t>::max());
  if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    const HGCSiliconDetId hid(id);
    auto ddd = get_ddd(geom, hid);
    size = ddd->cellSizeHex(hid.type());
  } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(HFNose)) {
    const HFNoseDetId hid(id);
    auto ddd = get_ddd(geom, hid);
    size = ddd->cellSizeHex(hid.type());
  } else if (id.det() == DetId::Forward) {
    const HGCalDetId hid(id);
    auto ddd = get_ddd(geom, hid);
    size = ddd->cellSizeHex(hid.waferType());
  } else {
    edm::LogError("getRadiusToSide::InvalidSiliconDetid")
        << "det id: " << std::hex << id.rawId() << std::dec << ":" << id.det() << " is not HGCal silicon!";
  }
  return size;
}

unsigned int RecHitTools::getLayer(const ForwardSubdetector type) const {
  int layer(0);
  switch (type) {
    case (ForwardSubdetector::HGCEE): {
      auto geomEE =
          static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HGCEE));
      layer = (geomEE->topology().dddConstants()).layers(true);
      break;
    }
    case (ForwardSubdetector::HGCHEF): {
      auto geomFH =
          static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HGCHEF));
      layer = (geomFH->topology().dddConstants()).layers(true);
      break;
    }
    case (ForwardSubdetector::HGCHEB): {
      auto geomBH =
          static_cast<const HcalGeometry*>(geom_->getSubdetectorGeometry(DetId::Hcal, HcalSubdetector::HcalEndcap));
      layer = (geomBH->topology().dddConstants())->getMaxDepth(1);
      break;
    }
    case (ForwardSubdetector::ForwardEmpty): {
      auto geomEE =
          static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HGCEE));
      layer = (geomEE->topology().dddConstants()).layers(true);
      auto geomFH =
          static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HGCHEF));
      layer += (geomFH->topology().dddConstants()).layers(true);
      auto geomBH =
          static_cast<const HcalGeometry*>(geom_->getSubdetectorGeometry(DetId::Hcal, HcalSubdetector::HcalEndcap));
      if (geomBH)
        layer += (geomBH->topology().dddConstants())->getMaxDepth(1);
      auto geomBH2 =
          static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HGCHEB));
      if (geomBH2)
        layer += (geomBH2->topology().dddConstants()).layers(true);
      break;
    }
    case (ForwardSubdetector::HFNose): {
      auto geomHFN =
          static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HFNose));
      layer = (geomHFN->topology().dddConstants()).layers(true);
      break;
    }
    default:
      layer = 0;
  }
  return (unsigned int)(layer);
}

unsigned int RecHitTools::getLayer(const DetId::Detector type, bool nose) const {
  int layer;
  switch (type) {
    case (DetId::HGCalEE): {
      auto geomEE =
          static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(type, ForwardSubdetector::ForwardEmpty));
      layer = (geomEE->topology().dddConstants()).layers(true);
      break;
    }
    case (DetId::HGCalHSi): {
      auto geomFH =
          static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(type, ForwardSubdetector::ForwardEmpty));
      layer = (geomFH->topology().dddConstants()).layers(true);
      break;
    }
    case (DetId::HGCalHSc): {
      auto geomBH =
          static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(type, ForwardSubdetector::ForwardEmpty));
      layer = (geomBH->topology().dddConstants()).layers(true);
      break;
    }
    case (DetId::Forward): {
      if (nose) {
        auto geomHFN = static_cast<const HGCalGeometry*>(
            geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HFNose));
        layer = (geomHFN->topology().dddConstants()).layers(true);
      } else {
        auto geomEE = static_cast<const HGCalGeometry*>(
            geom_->getSubdetectorGeometry(DetId::HGCalEE, ForwardSubdetector::ForwardEmpty));
        layer = (geomEE->topology().dddConstants()).layers(true);
        auto geomFH = static_cast<const HGCalGeometry*>(
            geom_->getSubdetectorGeometry(DetId::HGCalHSi, ForwardSubdetector::ForwardEmpty));
        layer += (geomFH->topology().dddConstants()).layers(true);
      }
      break;
    }
    default:
      layer = 0;
  }
  return (unsigned int)(layer);
}

unsigned int RecHitTools::getLayer(const DetId& id) const {
  unsigned int layer = std::numeric_limits<unsigned int>::max();
  if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
    layer = HGCSiliconDetId(id).layer();
  } else if (id.det() == DetId::HGCalHSc) {
    layer = HGCScintillatorDetId(id).layer();
  } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(HFNose)) {
    layer = HFNoseDetId(id).layer();
  } else if (id.det() == DetId::Forward) {
    layer = HGCalDetId(id).layer();
  } else if (id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
    layer = HcalDetId(id).depth();
  }
  return layer;
}

unsigned int RecHitTools::getLayerWithOffset(const DetId& id) const {
  unsigned int layer = getLayer(id);
  if (id.det() == DetId::Forward && id.subdetId() == HGCHEF) {
    layer += fhOffset_;
  } else if (id.det() == DetId::HGCalHSi || id.det() == DetId::HGCalHSc) {
    // DetId::HGCalHSc hits include the offset w.r.t. EE already
    layer += fhOffset_;
  } else if (id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
    layer += bhOffset_;
  }
  // no need to add offset for HFnose
  return layer;
}

std::pair<int, int> RecHitTools::getWafer(const DetId& id) const {
  int waferU = std::numeric_limits<int>::max();
  int waferV = 0;
  if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
    waferU = HGCSiliconDetId(id).waferU();
    waferV = HGCSiliconDetId(id).waferV();
  } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(HFNose)) {
    waferU = HFNoseDetId(id).waferU();
    waferV = HFNoseDetId(id).waferV();
  } else if (id.det() == DetId::Forward) {
    waferU = HGCalDetId(id).wafer();
  } else {
    edm::LogError("getWafer::InvalidSiliconDetid")
        << "det id: " << std::hex << id.rawId() << std::dec << ":" << id.det() << " is not HGCal silicon!";
  }
  return std::pair<int, int>(waferU, waferV);
}

std::pair<int, int> RecHitTools::getCell(const DetId& id) const {
  int cellU = std::numeric_limits<int>::max();
  int cellV = 0;
  if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
    cellU = HGCSiliconDetId(id).cellU();
    cellV = HGCSiliconDetId(id).cellV();
  } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(HFNose)) {
    cellU = HFNoseDetId(id).cellU();
    cellV = HFNoseDetId(id).cellV();
  } else if (id.det() == DetId::Forward) {
    cellU = HGCalDetId(id).cell();
  } else {
    edm::LogError("getCell::InvalidSiliconDetid")
        << "det id: " << std::hex << id.rawId() << std::dec << ":" << id.det() << " is not HGCal silicon!";
  }
  return std::pair<int, int>(cellU, cellV);
}

bool RecHitTools::isHalfCell(const DetId& id) const {
  bool ishalf = false;
  if (id.det() == DetId::Forward) {
    HGCalDetId hid(id);
    auto geom = getSubdetectorGeometry(hid);
    auto ddd = get_ddd(geom, hid);
    const int waferType = ddd->waferTypeT(hid.waferType());
    return ddd->isHalfCell(waferType, hid.cell());
  }
  //new geometry is always false
  return ishalf;
}

int RecHitTools::getLayerType(const DetId& id) const {
  auto layer_number = getLayerWithOffset(id);
  auto thickness = getSiThickIndex(id);
  auto geomNose =
      static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HFNose));
  auto isNose = geomNose ? true : false;
  auto isEELayer = (layer_number <= lastLayerEE(isNose));
  auto isScint = isScintillator(id);
  int layerType = -1;

  if (isScint) {
    layerType = CE_H_SCINT;
  }
  if (isEELayer) {
    if (thickness == 0) {
      layerType = CE_E_120;
    } else if (thickness == 1) {
      layerType = CE_E_200;
    } else if (thickness == 2) {
      layerType = CE_E_300;
    }
  } else {
    if (thickness == 0) {
      layerType = CE_H_120;
    } else if (thickness == 1) {
      layerType = CE_H_200;
    } else if (thickness == 2) {
      layerType = CE_H_300;
    }
  }
  assert(layerType != -1);
  return layerType;
}

bool RecHitTools::isSilicon(const DetId& id) const {
  return (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi ||
          (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(HFNose)));
}

bool RecHitTools::isScintillator(const DetId& id) const { return id.det() == DetId::HGCalHSc; }

bool RecHitTools::isOnlySilicon(const unsigned int layer) const {
  // HFnose TODO
  bool isonlysilicon = (layer % bhLastLayer_) < bhOffset_;
  return isonlysilicon;
}

float RecHitTools::getEta(const GlobalPoint& position, const float& vertex_z) const {
  GlobalPoint corrected_position = GlobalPoint(position.x(), position.y(), position.z() - vertex_z);
  return corrected_position.eta();
}

float RecHitTools::getEta(const DetId& id, const float& vertex_z) const {
  GlobalPoint position = getPosition(id);
  float eta = getEta(position, vertex_z);
  return eta;
}

float RecHitTools::getPhi(const GlobalPoint& position) const {
  float phi = atan2(position.y(), position.x());
  return phi;
}

float RecHitTools::getPhi(const DetId& id) const {
  GlobalPoint position = getPosition(id);
  float phi = atan2(position.y(), position.x());
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

std::pair<uint32_t, uint32_t> RecHitTools::firstAndLastLayer(DetId::Detector det, int subdet) const {
  if ((det == DetId::HGCalEE) || ((det == DetId::Forward) && (subdet == HGCEE))) {
    return std::make_pair(eeOffset_ + 1, fhOffset_);
  } else if ((det == DetId::HGCalHSi) || ((det == DetId::Forward) && (subdet == HGCHEF))) {
    return std::make_pair(fhOffset_ + 1, fhLastLayer_);
  } else if ((det == DetId::Forward) && (subdet == HFNose)) {
    return std::make_pair(1, noseLastLayer_);
  } else {
    return std::make_pair(bhFirstLayer_, bhLastLayer_);
  }
}

bool RecHitTools::maskCell(const DetId& id, int corners) const {
  if (id.det() == DetId::Hcal) {
    return false;
  } else {
    auto hg = static_cast<const HGCalGeometry*>(getSubdetectorGeometry(id));
    return hg->topology().dddConstants().maskCell(id, corners);
  }
}
