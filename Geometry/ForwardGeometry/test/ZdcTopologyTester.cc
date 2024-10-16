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
#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

class ZdcTopologyTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit ZdcTopologyTester(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void doTest(const ZdcTopology& topology);

  // ----------member data ---------------------------
  const edm::ESGetToken<ZdcTopology, HcalRecNumberingRecord> tokTopo_;
};

ZdcTopologyTester::ZdcTopologyTester(const edm::ParameterSet&)
    : tokTopo_{esConsumes<ZdcTopology, HcalRecNumberingRecord>(edm::ESInputTag{})} {}

void ZdcTopologyTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.add("zdcTopologyTester", desc);
}

void ZdcTopologyTester::analyze(edm::Event const&, edm::EventSetup const& iSetup) { doTest(iSetup.getData(tokTopo_)); }

void ZdcTopologyTester::doTest(const ZdcTopology& topology) {
  // Total number of valid cells
  for (int idet = 0; idet < 4; idet++) {
    int ndet(0);
    std::string det = "EM";
    HcalZDCDetId::Section section = HcalZDCDetId::EM;
    if (idet == 1) {
      det = "HAD";
      section = HcalZDCDetId::HAD;
    } else if (idet == 2) {
      det = "LUM";
      section = HcalZDCDetId::LUM;
    } else if (idet == 3) {
      det = "RPD";
      section = HcalZDCDetId::RPD;
    }
    for (int depth = 1; depth <= HcalZDCDetId::kDepTot; ++depth) {
      for (int zside = 0; zside <= 1; ++zside) {
        bool forward = (zside == 0) ? true : false;
        const HcalZDCDetId id(section, forward, depth);
        if (topology.valid(id))
          ++ndet;
      }
    }
    edm::LogVerbatim("HCalGeom") << "Number of valid cells in " << det << ": " << ndet;
  }

  // First test on movements along eta/phi directions
  edm::LogVerbatim("HCalGeom") << "\nTest on movements along transverse/longiudnal directions"
                               << "\n========================================================";
  for (int idet = 0; idet < 4; idet++) {
    HcalZDCDetId::Section section = HcalZDCDetId::EM;
    if (idet == 1)
      section = HcalZDCDetId::HAD;
    else if (idet == 2)
      section = HcalZDCDetId::LUM;
    else if (idet == 3)
      section = HcalZDCDetId::RPD;
    for (int depth = 1; depth <= HcalZDCDetId::kDepTot; ++depth) {
      for (int zside = 0; zside <= 1; ++zside) {
        bool forward = (zside == 0) ? true : false;
        const HcalZDCDetId id(section, forward, depth);
        if (topology.valid(id)) {
          std::vector<DetId> idT = topology.transverse(id);
          std::vector<DetId> idL = topology.longitudinal(id);
          edm::LogVerbatim("HCalGeom") << "Neighbours for : Tower " << id;
          std::ostringstream st1;
          st1 << "          " << idT.size() << " sets transverse:";
          for (auto& i : idT)
            st1 << " " << (HcalZDCDetId)(i());
          edm::LogVerbatim("HCalGeom") << st1.str();
          std::ostringstream st2;
          st2 << "          " << idL.size() << " sets along Longitunal:";
          for (auto& i : idL)
            st2 << " " << (HcalZDCDetId)(i());
          edm::LogVerbatim("HCalGeom") << st2.str();
        }
      }
    }
  }

  // Check on Dense Index
  edm::LogVerbatim("HCalGeom") << "\nCheck on Dense Index"
                               << "\n=====================";
  for (int idet = 0; idet < 4; idet++) {
    HcalZDCDetId::Section section = HcalZDCDetId::EM;
    if (idet == 1)
      section = HcalZDCDetId::HAD;
    else if (idet == 2)
      section = HcalZDCDetId::LUM;
    else if (idet == 3)
      section = HcalZDCDetId::RPD;
    for (int depth = 1; depth <= HcalZDCDetId::kDepTot; ++depth) {
      for (int zside = 0; zside <= 1; ++zside) {
        bool forward = (zside == 0) ? true : false;
        HcalZDCDetId cell(section, forward, depth);
        if (topology.valid(cell)) {
          unsigned int dense = topology.detId2DenseIndex(DetId(cell));
          DetId id = topology.denseId2detId(dense);
          std::string cherr = (cell.rawId() != id.rawId()) ? " **** ERROR *****" : "";
          edm::LogVerbatim("HCalGeom") << cell << " Dense " << std::hex << dense << std::dec << " o/p "
                                       << HcalZDCDetId(id) << cherr;
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZdcTopologyTester);
