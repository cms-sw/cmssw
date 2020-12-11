// -*- C++ -*-
//
// Package:    CaloPropagationTest
// Class:      CaloPropagationTest
//
/**\class CaloPropagationTest CaloPropagationTest.cc test/CaloPropagationTest.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class CaloPropagationTest : public edm::one::EDAnalyzer<> {
public:
  explicit CaloPropagationTest(const edm::ParameterSet&);
  ~CaloPropagationTest() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;
  const edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> tok_ctmap_;
};

CaloPropagationTest::CaloPropagationTest(const edm::ParameterSet&)
    : tok_geom_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      tok_magField_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      tok_ctmap_(esConsumes<CaloTowerConstituentsMap, CaloGeometryRecord>()) {}

CaloPropagationTest::~CaloPropagationTest() {}

// ------------ method called to produce the data  ------------
void CaloPropagationTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);
  const HcalGeometry* gHB = (const HcalGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
  const MagneticField* bField = &iSetup.getData(tok_magField_);
  const CaloTowerConstituentsMap* ctmap = &iSetup.getData(tok_ctmap_);

  const std::vector<DetId>& ids = gHB->getValidDetIds(DetId::Hcal, 0);
  bool debug(false);
  for (const auto& id : ids) {
    if (id.det() == DetId::Hcal && ((id.subdetId() == HcalBarrel) || (id.subdetId() == HcalEndcap))) {
      const HcalDetId hid(id);
      std::pair<DetId, bool> info = spr::propagateIdECAL(hid, geo, bField, debug);
      if (!info.second) {
        std::cout << "No valid Ecal Id found for " << hid << std::endl;
      } else {
        CaloTowerDetId tower = ctmap->towerOf(id);
        std::vector<DetId> idts = ctmap->constituentsOf(tower);
        std::string found("not found");
        for (auto idt : idts) {
          if (info.first == idt) {
            found = "found";
            break;
          }
        }
        if ((info.first).subdetId() == EcalBarrel) {
          std::cout << "Find " << EBDetId(info.first) << " as partner of " << hid << " and mtaching with tower "
                    << found << std::endl;
        } else {
          std::cout << "Find " << EEDetId(info.first) << " as partner of " << hid << " and mtaching with tower "
                    << found << std::endl;
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloPropagationTest);
