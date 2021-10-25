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
#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"
#include "Geometry/CaloTopology/interface/FastTimeTopology.h"

class FastTimeTopologyTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit FastTimeTopologyTester(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void doTest(const FastTimeTopology& topology);

  // ----------member data ---------------------------
  const edm::ESGetToken<FastTimeTopology, IdealGeometryRecord> tokTopo_;
};

FastTimeTopologyTester::FastTimeTopologyTester(const edm::ParameterSet&)
    : tokTopo_{esConsumes<FastTimeTopology, IdealGeometryRecord>(edm::ESInputTag{"", "FastTimeBarrel"})} {}

void FastTimeTopologyTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void FastTimeTopologyTester::analyze(edm::Event const&, edm::EventSetup const& iSetup) {
  doTest(iSetup.getData(tokTopo_));
}

void FastTimeTopologyTester::doTest(const FastTimeTopology& topology) {
  for (int izz = 0; izz <= 1; izz++) {
    int iz = (2 * izz - 1);
    for (int eta = 1; eta <= 265; ++eta) {
      for (int phi = 1; phi <= 720; ++phi) {
        const FastTimeDetId id(1, eta, phi, iz);
        if (topology.valid(id)) {
          std::cout << "Neighbours for Tower " << id << std::endl;
          std::vector<DetId> idE = topology.east(id);
          std::vector<DetId> idW = topology.west(id);
          std::vector<DetId> idN = topology.north(id);
          std::vector<DetId> idS = topology.south(id);
          std::cout << "          " << idE.size() << " sets along East:";
          for (auto& i : idE)
            std::cout << " " << static_cast<FastTimeDetId>(i());
          std::cout << std::endl;
          std::cout << "          " << idW.size() << " sets along West:";
          for (auto& i : idW)
            std::cout << " " << static_cast<FastTimeDetId>(i());
          std::cout << std::endl;
          std::cout << "          " << idN.size() << " sets along North:";
          for (auto& i : idN)
            std::cout << " " << static_cast<FastTimeDetId>(i());
          std::cout << std::endl;
          std::cout << "          " << idS.size() << " sets along South:";
          for (auto& i : idS)
            std::cout << " " << static_cast<FastTimeDetId>(i());
          std::cout << std::endl;
        }
        phi += 10;
      }
      eta += 5;
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(FastTimeTopologyTester);
