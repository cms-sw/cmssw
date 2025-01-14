#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include <iostream>
#include <string>

class HcalCellKount : public edm::one::EDAnalyzer<> {
public:
  explicit HcalCellKount(const edm::ParameterSet&);
  ~HcalCellKount(void) override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const int verbose_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
};

HcalCellKount::HcalCellKount(const edm::ParameterSet& iConfig) : verbose_(iConfig.getParameter<int>("Verbosity")) {
  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
}

void HcalCellKount::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("Verbosity", 0);
  descriptions.add("hcalCellKount", desc);
}

void HcalCellKount::analyze(edm::Event const& /*iEvent*/, const edm::EventSetup& iSetup) {
  const CaloGeometry *geo = &iSetup.getData(tok_geom_);

  // ECAL
  const CaloSubdetectorGeometry* bGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  if (bGeom != nullptr) 
    edm::LogVerbatim("HCalGeom") << "Valid ID for EcalBarrel: " << bGeom->getValidDetIds(DetId::Ecal, EcalBarrel).size();
  else
    edm::LogVerbatim("HCalGeom") << "EB Geometry does not exist";
  const CaloSubdetectorGeometry* eGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  if (eGeom != nullptr) 
    edm::LogVerbatim("HCalGeom") << "Valid ID for EcalEndcap: " << eGeom->getValidDetIds(DetId::Ecal, EcalEndcap).size();
  else
    edm::LogVerbatim("HCalGeom") << "EE Geometry does not exist";
  const CaloSubdetectorGeometry* sGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  if (sGeom != nullptr) 
    edm::LogVerbatim("HCalGeom") << "Valid ID for EcalPreshower: " << sGeom->getValidDetIds(DetId::Ecal, EcalPreshower).size();
  else
    edm::LogVerbatim("HCalGeom") << "ES Geometry does not exist";
  const CaloSubdetectorGeometry* tGeom = geo->getSubdetectorGeometry(DetId::Ecal, EcalTriggerTower);
  if (tGeom != nullptr) 
    edm::LogVerbatim("HCalGeom") << "Valid ID for EcalTriggerTower: " << tGeom->getValidDetIds(DetId::Ecal, EcalTriggerTower).size();
  else
    edm::LogVerbatim("HCalGeom") << "EcalTriggerTower Geometry does not exist";

  //HCAL
  const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  if (gHB != nullptr) {
    edm::LogVerbatim("HCalGeom") << "Valid ID for HcalBarrel: " << gHB->getValidDetIds(DetId::Hcal, HcalBarrel).size();
    edm::LogVerbatim("HCalGeom") << "Valid ID for HcalEndcap: " << gHB->getValidDetIds(DetId::Hcal, HcalEndcap).size();
    edm::LogVerbatim("HCalGeom") << "Valid ID for HcalOuter: " << gHB->getValidDetIds(DetId::Hcal, HcalOuter).size();
    edm::LogVerbatim("HCalGeom") << "Valid ID for HcalForward: " << gHB->getValidDetIds(DetId::Hcal, HcalForward).size();
    edm::LogVerbatim("HCalGeom") << "Valid ID for HcalTriggerTower: " << gHB->getValidDetIds(DetId::Hcal, HcalTriggerTower).size();
  } else {
    edm::LogVerbatim("HCalGeom") << "HCAL Geometry does not exist";
  }
  
  //HGCAL
  const CaloSubdetectorGeometry* gHGEE = geo->getSubdetectorGeometry(DetId::HGCalEE, 0);
  if (gHGEE != nullptr)
    edm::LogVerbatim("HCalGeom") << "Valid ID for HGCalEE: " << gHGEE->getValidDetIds(DetId::HGCalEE, 0).size();
  else
    edm::LogVerbatim("HCalGeom") << "HGCaLEE Geometry does not exist";
  const CaloSubdetectorGeometry* gHGHSi = geo->getSubdetectorGeometry(DetId::HGCalHSi, 0);
  if (gHGHSi != nullptr)
    edm::LogVerbatim("HCalGeom") << "Valid ID for HGCalHSi: " << gHGHSi->getValidDetIds(DetId::HGCalHSi, 0).size();
  else
    edm::LogVerbatim("HCalGeom") << "HGCaLHSi Geometry does not exist";
  const CaloSubdetectorGeometry* gHGHSc = geo->getSubdetectorGeometry(DetId::HGCalHSc, 0);
  if (gHGHSc != nullptr)
    edm::LogVerbatim("HCalGeom") << "Valid ID for HGCalHSc: " << gHGHSc->getValidDetIds(DetId::HGCalHSc, 0).size();
  else
    edm::LogVerbatim("HCalGeom") << "HGCaLHSc Geometry does not exist";
}

DEFINE_FWK_MODULE(HcalCellKount);
