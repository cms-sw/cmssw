#include <iostream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

class CaloTowerTopologyTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit CaloTowerTopologyTester(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void doTest(const CaloTowerTopology& topology);

  // ----------member data ---------------------------
  const edm::ESGetToken<CaloTowerTopology, HcalRecNumberingRecord> tokTopo_;
};

CaloTowerTopologyTester::CaloTowerTopologyTester(const edm::ParameterSet&)
    : tokTopo_{esConsumes<CaloTowerTopology, HcalRecNumberingRecord>(edm::ESInputTag{})} {}

void CaloTowerTopologyTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void CaloTowerTopologyTester::analyze(edm::Event const&, edm::EventSetup const& iSetup) {
  doTest(iSetup.getData(tokTopo_));
}

void CaloTowerTopologyTester::doTest(const CaloTowerTopology& topology) {
  for (int ieta = -topology.lastHFRing(); ieta <= topology.lastHFRing(); ieta++) {
    for (int iphi = 1; iphi <= 72; iphi++) {
      const CaloTowerDetId id(ieta, iphi);
      if (topology.validDetId(id)) {
        std::vector<DetId> idE = topology.east(id);
        std::vector<DetId> idW = topology.west(id);
        std::vector<DetId> idN = topology.north(id);
        std::vector<DetId> idS = topology.south(id);
        std::cout << "Neighbours for : Tower " << id << std::endl;
        std::cout << "          " << idE.size() << " sets along East:";
        for (auto& i : idE)
          std::cout << " " << static_cast<CaloTowerDetId>(i());
        std::cout << std::endl;
        std::cout << "          " << idW.size() << " sets along West:";
        for (auto& i : idW)
          std::cout << " " << static_cast<CaloTowerDetId>(i());
        std::cout << std::endl;
        std::cout << "          " << idN.size() << " sets along North:";
        for (auto& i : idN)
          std::cout << " " << static_cast<CaloTowerDetId>(i());
        std::cout << std::endl;
        std::cout << "          " << idS.size() << " sets along South:";
        for (auto& i : idS)
          std::cout << " " << static_cast<CaloTowerDetId>(i());
        std::cout << std::endl;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloTowerTopologyTester);
