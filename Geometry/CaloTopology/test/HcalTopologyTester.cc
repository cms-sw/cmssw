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

#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class HcalTopologyTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HcalTopologyTester(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void doTest(const HcalTopology& topology);

  // ----------member data ---------------------------
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tokTopo_;
};

HcalTopologyTester::HcalTopologyTester(const edm::ParameterSet&)
    : tokTopo_{esConsumes<HcalTopology, HcalRecNumberingRecord>(edm::ESInputTag{})} {}

void HcalTopologyTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HcalTopologyTester::analyze(edm::Event const&, edm::EventSetup const& iSetup) { doTest(iSetup.getData(tokTopo_)); }

void HcalTopologyTester::doTest(const HcalTopology& topology) {
  // First test on movements along eta/phi directions
  std::cout << "\nTest on movements along eta/phi directions" << std::endl
            << "==========================================" << std::endl;
  for (int idet = 0; idet < 4; idet++) {
    HcalSubdetector subdet = HcalBarrel;
    if (idet == 1)
      subdet = HcalOuter;
    else if (idet == 2)
      subdet = HcalEndcap;
    else if (idet == 3)
      subdet = HcalForward;
    for (int depth = 1; depth < 4; ++depth) {
      for (int ieta = -41; ieta <= 41; ieta++) {
        for (int iphi = 1; iphi <= 72; iphi++) {
          const HcalDetId id(subdet, ieta, iphi, depth);
          if (topology.valid(id)) {
            std::vector<DetId> idE = topology.east(id);
            std::vector<DetId> idW = topology.west(id);
            std::vector<DetId> idN = topology.north(id);
            std::vector<DetId> idS = topology.south(id);
            std::vector<DetId> idU = topology.up(id);
            std::cout << "Neighbours for : Tower " << id << std::endl;
            std::cout << "          " << idE.size() << " sets along East:";
            for (auto& i : idE)
              std::cout << " " << (HcalDetId)(i());
            std::cout << std::endl;
            std::cout << "          " << idW.size() << " sets along West:";
            for (auto& i : idW)
              std::cout << " " << (HcalDetId)(i());
            std::cout << std::endl;
            std::cout << "          " << idN.size() << " sets along North:";
            for (auto& i : idN)
              std::cout << " " << (HcalDetId)(i());
            std::cout << std::endl;
            std::cout << "          " << idS.size() << " sets along South:";
            for (auto& i : idS)
              std::cout << " " << (HcalDetId)(i());
            std::cout << std::endl;
            std::cout << "          " << idU.size() << " sets up in depth:";
            for (auto& i : idU)
              std::cout << " " << (HcalDetId)(i());
            std::cout << std::endl;
          }
        }
      }
    }
  }

  // Check on Dense Index
  std::cout << "\nCheck on Dense Index" << std::endl << "=====================" << std::endl;
  int maxDepthHB = topology.maxDepthHB();
  int maxDepthHE = topology.maxDepthHE();
  for (int det = 1; det <= HcalForward; det++) {
    for (int eta = -HcalDetId::kHcalEtaMask2; eta <= (int)(HcalDetId::kHcalEtaMask2); eta++) {
      for (unsigned int phi = 0; phi <= HcalDetId::kHcalPhiMask2; phi++) {
        for (int depth = 1; depth < maxDepthHB + maxDepthHE; depth++) {
          HcalDetId cell((HcalSubdetector)det, eta, phi, depth);
          if (topology.valid(cell)) {
            unsigned int dense = topology.detId2denseId(DetId(cell));
            DetId id = topology.denseId2detId(dense);
            if (cell == HcalDetId(id))
              std::cout << cell << " Dense " << std::hex << dense << std::dec << " o/p " << HcalDetId(id) << std::endl;
            else
              std::cout << cell << " Dense " << std::hex << dense << std::dec << " o/p " << HcalDetId(id)
                        << " **** ERROR *****" << std::endl;
          }
        }
      }
    }
  }

  // Check list of depths
  std::cout << "\nCheck list of Depths" << std::endl << "====================" << std::endl;
  for (int eta = topology.lastHERing() - 2; eta <= topology.lastHERing(); ++eta) {
    for (unsigned int phi = 0; phi <= HcalDetId::kHcalPhiMask2; phi++) {
      for (int depth = 1; depth <= maxDepthHE; depth++) {
        HcalDetId cell(HcalEndcap, eta, phi, depth);
        if (topology.valid(cell)) {
          std::vector<int> depths = topology.mergedDepthList29(cell);
          std::cout << cell << " is with merge depth flag " << topology.mergedDepth29(cell) << " having "
                    << depths.size() << " merged depths:";
          for (unsigned int k = 0; k < depths.size(); ++k)
            std::cout << " [" << k << "]:" << depths[k];
          std::cout << std::endl;
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalTopologyTester);
