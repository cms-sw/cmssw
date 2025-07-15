#include <iostream>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "CoralBase/Exception.h"

class HGCalTestRecHitTool : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalTestRecHitTool(const edm::ParameterSet&);
  ~HGCalTestRecHitTool() override;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  std::vector<double> retrieveLayerPositions(unsigned int);
  template <typename GEOM>
  void check_geom(const GEOM* geom) const;
  template <typename DDD>
  void check_ddd(const DDD* ddd) const;
  const HGCalDDDConstants* get_ddd(const DetId& detid) const;
  const HcalDDDRecConstants* get_ddd(const HcalDetId& detid) const;
  double getLayerZ(DetId const&) const;
  double getLayerZ(int type, int layer) const;
  GlobalPoint getPosition(DetId const&) const;
  int getScintMaxIphi(const DetId& id) const;
  bool isScintillatorFine(const DetId& id) const;

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  const CaloGeometry* geom_;
  const int mode_;
  int layerEE_, layerFH_, layerBH_;
  int eeOffset_, fhOffset_, bhOffset_;
  int layerEE1000_, layerFH1000_, layerBH1000_;
};

HGCalTestRecHitTool::HGCalTestRecHitTool(const edm::ParameterSet& iC)
    : geomToken_{esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{})},
      geom_(nullptr),
      mode_(iC.getParameter<int>("Mode")) {
  layerEE_ = layerFH_ = layerBH_ = 0;
  eeOffset_ = fhOffset_ = bhOffset_ = 0;
  layerEE1000_ = layerFH1000_ = layerBH1000_ = 0;
  edm::LogVerbatim("HGCalGeom") << "Instantiate HGCalTestRecHitTool with mode " << mode_;
}

HGCalTestRecHitTool::~HGCalTestRecHitTool() {}

void HGCalTestRecHitTool::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  if (auto pG = iSetup.getHandle(geomToken_)) {
    geom_ = pG.product();
    auto geomEE = ((mode_ == 0) ? static_cast<const HGCalGeometry*>(
                                      geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HGCEE))
                                : static_cast<const HGCalGeometry*>(
                                      geom_->getSubdetectorGeometry(DetId::HGCalEE, ForwardSubdetector::ForwardEmpty)));
    layerEE_ = (geomEE->topology().dddConstants()).layers(true);
    eeOffset_ = (geomEE->topology().dddConstants()).getLayerOffset();
    layerEE1000_ = (geomEE->topology().dddConstants()).getLayer(10000., true);
    edm::LogVerbatim("HGCalGeom") << "EE::Layers " << layerEE_ << " Offset " << eeOffset_ << " Layer # at 1000 cm "
                                  << layerEE1000_;
    auto geomFH = ((mode_ == 0) ? static_cast<const HGCalGeometry*>(
                                      geom_->getSubdetectorGeometry(DetId::Forward, ForwardSubdetector::HGCHEF))
                                : static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(
                                      DetId::HGCalHSi, ForwardSubdetector::ForwardEmpty)));
    layerFH_ = (geomFH->topology().dddConstants()).layers(true);
    fhOffset_ = (geomFH->topology().dddConstants()).getLayerOffset();
    layerFH1000_ = (geomFH->topology().dddConstants()).getLayer(10000., true);
    edm::LogVerbatim("HGCalGeom") << "FH::Layers " << layerFH_ << " Offsets " << fhOffset_ << " Layer # at 1000 cm "
                                  << layerFH1000_;
    if (mode_ == 0) {
      auto geomBH =
          static_cast<const HcalGeometry*>(geom_->getSubdetectorGeometry(DetId::Hcal, HcalSubdetector::HcalEndcap));
      layerBH_ = (geomBH->topology().dddConstants())->getMaxDepth(1);
    } else {
      auto geomBH = static_cast<const HGCalGeometry*>(
          geom_->getSubdetectorGeometry(DetId::HGCalHSc, ForwardSubdetector::ForwardEmpty));
      layerBH_ = (geomBH->topology().dddConstants()).layers(true);
      bhOffset_ = (geomBH->topology().dddConstants()).getLayerOffset();
      layerBH1000_ = (geomBH->topology().dddConstants()).getLayer(10000., true);
    }
    edm::LogVerbatim("HGCalGeom") << "BH::Layers " << layerBH_ << " nOffsets " << bhOffset_ << " Layer # at 1000 cm "
                                  << layerBH1000_;
    for (int layer = 1; layer <= layerEE_; ++layer)
      edm::LogVerbatim("HGCalGeom") << "EE Layer " << layer << " Wafers "
                                    << (geomEE->topology().dddConstants()).wafers(layer, 0) << ":"
                                    << (geomEE->topology().dddConstants()).wafers(layer, 1) << ":"
                                    << (geomEE->topology().dddConstants()).wafers(layer, 2);
    for (int layer = 1; layer <= layerFH_; ++layer)
      edm::LogVerbatim("HGCalGeom") << "FH Layer " << layer << " Wafers "
                                    << (geomFH->topology().dddConstants()).wafers(layer, 0) << ":"
                                    << (geomFH->topology().dddConstants()).wafers(layer, 1) << ":"
                                    << (geomFH->topology().dddConstants()).wafers(layer, 2);
    if (mode_ != 0) {
      auto geomBH = static_cast<const HGCalGeometry*>(
          geom_->getSubdetectorGeometry(DetId::HGCalHSc, ForwardSubdetector::ForwardEmpty));
      int firstL = (geomBH->topology().dddConstants()).firstLayer();
      edm::LogVerbatim("HGCalGeom") << "BH First Layer " << firstL << " Total " << layerBH_;
      for (int lay = 1; lay <= layerBH_; ++lay) {
        int layer = firstL + lay - 1;
        int ring = (geomBH->topology().dddConstants()).tileRings(layer).first + 1;
        auto typm = (geomBH->topology().dddConstants()).tileType(layer, ring, 1);
        HGCScintillatorDetId id(typm.first, layer, ring, 1, false, typm.second, 0);
        edm::LogVerbatim("HGCalGeom") << "BH Layer " << layer << " Ring " << ring << " Max phi "
                                      << getScintMaxIphi(DetId(id)) << " Fine " << isScintillatorFine(DetId(id));
      }
    }
    int nlayer = ((mode_ == 0) ? (layerEE_ + layerFH_ + layerBH_) : (layerEE_ + layerFH_));
    retrieveLayerPositions(nlayer);
  } else {
    edm::LogVerbatim("HGCalGeom") << "Cannot get valid CaloGeometry Object" << std::endl;
  }
}

template <typename GEOM>
void HGCalTestRecHitTool::check_geom(const GEOM* geom) const {
  if (nullptr == geom) {
    throw cms::Exception("HGCalTestRecHitTools") << "Geometry not provided yet to HGCalTestRecHitTools!";
  }
}

template <typename DDD>
void HGCalTestRecHitTool::check_ddd(const DDD* ddd) const {
  if (nullptr == ddd) {
    throw cms::Exception("HGCalTestRecHitTools") << "DDDConstants not accessibl to HGCalTestRecHitTools!";
  }
}

std::vector<double> HGCalTestRecHitTool::retrieveLayerPositions(unsigned layers) {
  DetId id;
  std::vector<double> layerPositions(layers);
  for (int ilayer = 1; ilayer <= (int)(layers); ++ilayer) {
    int lay(ilayer), type(0);
    if (ilayer <= layerEE_) {
      id = ((mode_ == 0) ? static_cast<DetId>(HGCalDetId(ForwardSubdetector::HGCEE, 1, lay, 1, 2, 1))
                         : static_cast<DetId>(HGCSiliconDetId(DetId::HGCalEE, 1, 0, lay, 3, 3, 1, 1)));
    } else if (ilayer > layerEE_ && ilayer <= (layerEE_ + layerFH_)) {
      lay = ilayer - layerEE_;
      id = ((mode_ == 0) ? static_cast<DetId>(HGCalDetId(ForwardSubdetector::HGCHEF, 1, lay, 1, 2, 1))
                         : static_cast<DetId>(HGCSiliconDetId(DetId::HGCalHSi, 1, 0, lay, 3, 3, 1, 1)));
      type = 1;
    } else {
      lay = ilayer - (layerEE_ + layerFH_);
      id = HcalDetId(HcalSubdetector::HcalEndcap, 50, 100, lay);
      type = 2;
    }
    const GlobalPoint pos = getPosition(id);
    if (id.det() == DetId::Hcal) {
      edm::LogVerbatim("HGCalGeom") << "GEOM  layer " << ilayer << " ID " << HcalDetId(id);
    } else if (id.det() == DetId::Forward) {
      edm::LogVerbatim("HGCalGeom") << "GEOM  layer " << ilayer << " ID " << HGCalDetId(id);
    } else if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
      edm::LogVerbatim("HGCalGeom") << "GEOM  layer " << ilayer << " ID " << HGCSiliconDetId(id);
    } else {
      edm::LogVerbatim("HGCalGeom") << "GEOM  layer " << ilayer << " ID " << HGCScintillatorDetId(id);
    }
    edm::LogVerbatim("HGCalGeom") << " Z " << pos.z() << ":" << getLayerZ(id) << ":" << getLayerZ(type, lay)
                                  << std::endl;
    layerPositions[ilayer - 1] = pos.z();
  }
  return layerPositions;
}

const HcalDDDRecConstants* HGCalTestRecHitTool::get_ddd(const HcalDetId& detid) const {
  auto geom = geom_->getSubdetectorGeometry(detid);
  check_geom(geom);
  const HcalGeometry* hc = static_cast<const HcalGeometry*>(geom);
  const HcalDDDRecConstants* ddd = hc->topology().dddConstants();
  check_ddd(ddd);
  return ddd;
}

const HGCalDDDConstants* HGCalTestRecHitTool::get_ddd(const DetId& detid) const {
  auto geom = geom_->getSubdetectorGeometry(detid);
  check_geom(geom);
  const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom);
  const HGCalDDDConstants* ddd = &(hg->topology().dddConstants());
  check_ddd(ddd);
  return ddd;
}

GlobalPoint HGCalTestRecHitTool::getPosition(const DetId& id) const {
  auto geom = geom_->getSubdetectorGeometry(id);
  check_geom(geom);
  GlobalPoint position;
  if (id.det() == DetId::Hcal) {
    position = geom->getGeometry(id)->getPosition();
  } else {
    position = (dynamic_cast<const HGCalGeometry*>(geom))->getPosition(id);
  }
  return position;
}

double HGCalTestRecHitTool::getLayerZ(const DetId& id) const {
  double zpos(0);
  if (id.det() == DetId::Hcal) {
    auto geom = geom_->getSubdetectorGeometry(id);
    check_geom(geom);
    zpos = (static_cast<const HcalGeometry*>(geom))->getGeometry(id)->getPosition().z();
  } else {
    const HGCalDDDConstants* ddd = get_ddd(id);
    int layer = ((id.det() == DetId::Forward) ? HGCalDetId(id).layer()
                                              : ((id.det() != DetId::HGCalHSc) ? HGCSiliconDetId(id).layer()
                                                                               : HGCScintillatorDetId(id).layer()));
    zpos = ddd->waferZ(layer, true);
  }
  return zpos;
}

double HGCalTestRecHitTool::getLayerZ(int type, int layer) const {
  double zpos(0);
  if (type == 2) {
    auto geom =
        static_cast<const HcalGeometry*>(geom_->getSubdetectorGeometry(DetId::Hcal, HcalSubdetector::HcalEndcap));
    check_geom(geom);
    auto ddd = (geom->topology().dddConstants());
    check_ddd(ddd);
    std::pair<int, int> etar = ddd->getEtaRange(1);
    zpos = ddd->getRZ(2, etar.second, layer);
  } else {
    auto geom = ((type == 1) ? ((mode_ == 0) ? static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(
                                                   DetId::Forward, ForwardSubdetector::HGCHEF))
                                             : static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(
                                                   DetId::HGCalHSi, ForwardSubdetector::ForwardEmpty)))
                             : ((mode_ == 0) ? static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(
                                                   DetId::Forward, ForwardSubdetector::HGCEE))
                                             : static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(
                                                   DetId::HGCalEE, ForwardSubdetector::ForwardEmpty))));
    check_geom(geom);
    auto ddd = &(geom->topology().dddConstants());
    check_ddd(ddd);
    zpos = ddd->waferZ(layer, true);
  }
  return zpos;
}

int HGCalTestRecHitTool::getScintMaxIphi(const DetId& id) const {
  if (id.det() == DetId::HGCalHSc) {
    int layer = HGCScintillatorDetId(id).layer();
    auto hg = static_cast<const HGCalGeometry*>(
        geom_->getSubdetectorGeometry(DetId::HGCalHSc, ForwardSubdetector::ForwardEmpty));
    return hg->topology().dddConstants().maxCells(layer, true);
  } else {
    return 0;
  }
}

bool HGCalTestRecHitTool::isScintillatorFine(const DetId& id) const {
  edm::LogVerbatim("HGCalGeom") << "isScintillatorFine " << id.det() << ":" << DetId::HGCalHSc << ":" << bhOffset_;
  if (id.det() == DetId::HGCalHSc) {
    int layer = HGCScintillatorDetId(id).layer();
    auto hg = static_cast<const HGCalGeometry*>(
        geom_->getSubdetectorGeometry(DetId::HGCalHSc, ForwardSubdetector::ForwardEmpty));
    return hg->topology().dddConstants().scintFine(layer);
  } else {
    return false;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTestRecHitTool);
