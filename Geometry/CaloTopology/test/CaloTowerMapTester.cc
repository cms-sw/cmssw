#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class CaloTowerMapTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit CaloTowerMapTester(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void doTest(const CaloGeometry* geo, const CaloTowerConstituentsMap* ctmap);

private:
  // ----------member data ---------------------------
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tokGeom_;
  const edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> tokMap_;
};

CaloTowerMapTester::CaloTowerMapTester(const edm::ParameterSet&)
    : tokGeom_{esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{})},
      tokMap_{esConsumes<CaloTowerConstituentsMap, CaloGeometryRecord>(edm::ESInputTag{})} {}

void CaloTowerMapTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.add("caloTowerMapTester", desc);
}

void CaloTowerMapTester::analyze(edm::Event const&, edm::EventSetup const& iSetup) {
  const CaloGeometry* geo = &iSetup.getData(tokGeom_);
  const CaloTowerConstituentsMap* ctmap = &iSetup.getData(tokMap_);
  doTest(geo, ctmap);
}

void CaloTowerMapTester::doTest(const CaloGeometry* geo, const CaloTowerConstituentsMap* ctmap) {
  const HcalGeometry* hgeo = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
  const std::vector<DetId>& dets = hgeo->getValidDetIds(DetId::Hcal, 0);

  for (const auto& id : dets) {
    CaloTowerDetId tower = ctmap->towerOf(id);
    std::vector<DetId> ids = ctmap->constituentsOf(tower);
    edm::LogVerbatim("CaloTower") << static_cast<HcalDetId>(id) << " belongs to " << tower << " which has "
                                  << ids.size() << " constituents\n";
    for (unsigned int i = 0; i < ids.size(); ++i) {
      std::ostringstream st1;
      st1 << "[" << i << "] " << std::hex << ids[i].rawId() << std::dec;
      if (ids[i].det() == DetId::Ecal && ids[i].subdetId() == EcalBarrel) {
        st1 << " " << static_cast<EBDetId>(ids[i]);
      } else if (ids[i].det() == DetId::Ecal && ids[i].subdetId() == EcalEndcap) {
        st1 << " " << static_cast<EEDetId>(ids[i]);
      } else if (ids[i].det() == DetId::Ecal && ids[i].subdetId() == EcalPreshower) {
        st1 << " " << static_cast<ESDetId>(ids[i]);
      } else if (ids[i].det() == DetId::Hcal) {
        st1 << " " << static_cast<HcalDetId>(ids[i]);
      }
      edm::LogVerbatim("CaloTower") << st1.str();
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloTowerMapTester);
