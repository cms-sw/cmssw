#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
  descriptions.add("hcalTopologyTester", desc);
}

void HcalTopologyTester::analyze(edm::Event const&, edm::EventSetup const& iSetup) { doTest(iSetup.getData(tokTopo_)); }

void HcalTopologyTester::doTest(const HcalTopology& topology) {
  // Total number of valid cells
  edm::LogVerbatim("HCalGeom") << "Total number of cells in HB:" << topology.getHBSize()
                               << " HE: " << topology.getHESize() << " HF: " << topology.getHFSize()
                               << " HO: " << topology.getHOSize() << " HT: " << topology.getHTSize()
                               << " Calib: " << topology.getCALIBSize() << " Overall: " << topology.ncells();
  std::vector<std::string> dets = {"HB", "HE", "HO", "HF"};
  for (int det = 1; det <= 4; ++det)
    edm::LogVerbatim("HCalGeom") << "Valid cells for " << dets[det - 1] << " = " << topology.ncells(det);

  // First test on movements along eta/phi directions
  edm::LogVerbatim("HCalGeom") << "\nTest on movements along eta/phi directions"
                               << "\n==========================================";
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
            edm::LogVerbatim("HCalGeom") << "Neighbours for : Tower " << id;
            std::ostringstream st1;
            st1 << "          " << idE.size() << " sets along East:";
            for (auto& i : idE)
              st1 << " " << (HcalDetId)(i());
            edm::LogVerbatim("HCalGeom") << st1.str();
            std::ostringstream st2;
            st2 << "          " << idW.size() << " sets along West:";
            for (auto& i : idW)
              st2 << " " << (HcalDetId)(i());
            edm::LogVerbatim("HCalGeom") << st2.str();
            std::ostringstream st3;
            st3 << "          " << idN.size() << " sets along North:";
            for (auto& i : idN)
              st3 << " " << (HcalDetId)(i());
            edm::LogVerbatim("HCalGeom") << st3.str();
            std::ostringstream st4;
            st4 << "          " << idS.size() << " sets along South:";
            for (auto& i : idS)
              st4 << " " << (HcalDetId)(i());
            edm::LogVerbatim("HCalGeom") << st4.str();
            std::ostringstream st5;
            st5 << "          " << idU.size() << " sets up in depth:";
            for (auto& i : idU)
              st5 << " " << (HcalDetId)(i());
            edm::LogVerbatim("HCalGeom") << st5.str();
          }
        }
      }
    }
  }

  // Check on Dense Index
  edm::LogVerbatim("HCalGeom") << "\nCheck on Dense Index"
                               << "\n=====================";
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
            std::string cherr = (cell != HcalDetId(id)) ? " **** ERROR *****" : "";
            edm::LogVerbatim("HCalGeom") << cell << " Dense " << std::hex << dense << std::dec << " o/p "
                                         << HcalDetId(id) << cherr;
          }
        }
      }
    }
  }

  // Check list of depths
  edm::LogVerbatim("HCalGeom") << "\nCheck list of Depths"
                               << "\n====================";
  for (int eta = topology.lastHERing() - 2; eta <= topology.lastHERing(); ++eta) {
    for (unsigned int phi = 0; phi <= HcalDetId::kHcalPhiMask2; phi++) {
      for (int depth = 1; depth <= maxDepthHE; depth++) {
        HcalDetId cell(HcalEndcap, eta, phi, depth);
        if (topology.valid(cell)) {
          std::vector<int> depths = topology.mergedDepthList29(cell);
          std::ostringstream st1;
          st1 << cell << " is with merge depth flag " << topology.mergedDepth29(cell) << " having " << depths.size()
              << " merged depths:";
          for (unsigned int k = 0; k < depths.size(); ++k)
            st1 << " [" << k << "]:" << depths[k];
          edm::LogVerbatim("HCalGeom") << st1.str();
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalTopologyTester);
