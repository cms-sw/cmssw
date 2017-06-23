#include <iostream>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
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
  explicit HGCalTestRecHitTool(const edm::ParameterSet& );
  ~HGCalTestRecHitTool();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
  
private:
  std::vector<double> retrieveLayerPositions(unsigned int);
  template<typename GEOM> void check_geom(const GEOM* geom) const;
  template<typename DDD>  void check_ddd(const DDD* ddd) const;
  const HGCalDDDConstants* get_ddd(const HGCalDetId& detid) const;
  const HcalDDDRecConstants* get_ddd(const HcalDetId& detid) const;
  double      getLayerZ(DetId const&) const;
  double      getLayerZ(int type, int layer) const;
  GlobalPoint getPosition(DetId const&) const;

  const CaloGeometry *geom_;
};

HGCalTestRecHitTool::HGCalTestRecHitTool(const edm::ParameterSet& iC) { }


HGCalTestRecHitTool::~HGCalTestRecHitTool() {}

void HGCalTestRecHitTool::analyze(const edm::Event& , 
				  const edm::EventSetup& iSetup) {

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);

  if (pG.isValid()) {
    geom_ = pG.product();
    retrieveLayerPositions(52);
  } else {
    std::cout << "Cannot get valid CaloGeometry Object" << std::endl;
  }
}

template<typename GEOM>
void HGCalTestRecHitTool::check_geom(const GEOM* geom) const {
  if (nullptr == geom) {
    throw cms::Exception("HGCalTestRecHitTools")
      << "Geometry not provided yet to HGCalTestRecHitTools!";
  }
}

template<typename DDD>
void HGCalTestRecHitTool::check_ddd(const DDD* ddd) const {
  if (nullptr == ddd) {
    throw cms::Exception("HGCalTestRecHitTools")
      << "DDDConstants not accessibl to HGCalTestRecHitTools!";
  }
}

std::vector<double> HGCalTestRecHitTool::retrieveLayerPositions(unsigned layers) {

  DetId id;
  std::vector<double> layerPositions;
  for (unsigned ilayer=1; ilayer<=layers; ++ilayer) {
    int lay(ilayer), type(0);
    if (ilayer<=28) {
      id  = HGCalDetId(ForwardSubdetector::HGCEE,1,lay,1,2,1);
    } else if (ilayer>28 && ilayer<=40) {
      lay = ilayer-28;
      id  = HGCalDetId(ForwardSubdetector::HGCHEF,1,lay,1,2,1);
      type= 1;
    } else {
      lay = ilayer-40;
      id  = HcalDetId(HcalSubdetector::HcalEndcap, 50, 100, lay);
      type= 2;
    }
    const GlobalPoint pos = getPosition(id);
    std::cout << "GEOM  layer " << ilayer;
    if (id.det() == DetId::Hcal) std::cout << " ID " << HcalDetId(id);
    else                         std::cout << " ID " <<HGCalDetId(id);
    std::cout << " Z " << pos.z() << ":" << getLayerZ(id) << ":"
	      << getLayerZ(type,lay) << std::endl;
    layerPositions.push_back(pos.z());
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

const HGCalDDDConstants* HGCalTestRecHitTool::get_ddd(const HGCalDetId& detid) const {
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
    const auto* hg = static_cast<const HGCalGeometry*>(geom);
    position = hg->getPosition(id);
  }
  return position;
}

double HGCalTestRecHitTool::getLayerZ(const DetId& id) const {

  double zpos(0);
  if (id.det() == DetId::Hcal) {
    auto geom = geom_->getSubdetectorGeometry(id);
    check_geom(geom);
    zpos = geom->getGeometry(id)->getPosition().z();
  } else {
    const HGCalDDDConstants* ddd = get_ddd(HGCalDetId(id));
    zpos = ddd->waferZ(HGCalDetId(id).layer(),true);
  }
  return zpos;
}

double HGCalTestRecHitTool::getLayerZ(int type, int layer) const {

  double zpos(0);
  if (type == 2) {
    auto geom = 
      static_cast<const HcalGeometry*>(geom_->getSubdetectorGeometry(DetId::Hcal,HcalSubdetector::HcalEndcap));
    check_geom(geom);
    auto ddd = (geom->topology().dddConstants());
    check_ddd(ddd);
    zpos = ddd->getRZ(2,layer+7);
  } else {
    auto geom = (type == 1) ? 
      static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCHEF)) :
      static_cast<const HGCalGeometry*>(geom_->getSubdetectorGeometry(DetId::Forward,ForwardSubdetector::HGCEE));
    check_geom(geom);
    auto ddd = &(geom->topology().dddConstants());
    check_ddd(ddd);
    zpos = ddd->waferZ(layer,true);
  }
  return zpos;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTestRecHitTool);
