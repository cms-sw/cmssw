#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include <iostream>

class HcalGeometryDetIdAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalGeometryDetIdAnalyzer(const edm::ParameterSet&);
  ~HcalGeometryDetIdAnalyzer(void) override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  bool useOld_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_ddrec_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;
};

HcalGeometryDetIdAnalyzer::HcalGeometryDetIdAnalyzer(const edm::ParameterSet& iConfig) {
  useOld_ = iConfig.getParameter<bool>("UseOldLoader");
  tok_ddrec_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord>();
  tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
}

HcalGeometryDetIdAnalyzer::~HcalGeometryDetIdAnalyzer(void) {}

void HcalGeometryDetIdAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const HcalDDDRecConstants hcons = iSetup.getData(tok_ddrec_);
  const HcalTopology topology = iSetup.getData(tok_htopo_);

  CaloSubdetectorGeometry* caloGeom(nullptr);
  if (useOld_) {
    HcalHardcodeGeometryLoader m_loader;
    caloGeom = m_loader.load(topology);
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader;
    caloGeom = m_loader.load(topology, hcons);
  }
  const std::vector<DetId>& ids = caloGeom->getValidDetIds();

  int counter = 0;
  for (std::vector<DetId>::const_iterator i = ids.begin(), iEnd = ids.end(); i != iEnd; ++i, ++counter) {
    HcalDetId hid = (*i);
    unsigned int did = topology.detId2denseId(*i);
    HcalDetId rhid = topology.denseId2detId(did);

    std::cout << counter << ": din " << std::hex << did << std::dec << ": " << hid << " == " << rhid << std::endl;
    assert(hid == rhid);
  }
  std::cout << "No error found among " << counter << " HCAL valid ID's\n";
}

DEFINE_FWK_MODULE(HcalGeometryDetIdAnalyzer);
