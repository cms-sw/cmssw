// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//calo headers
#include "CondFormats/HcalObjects/interface/HBHEDarkening.h"
#include "CondFormats/DataRecord/interface/HBHEDarkeningRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HBHERecalibration.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

//STL headers
#include <iostream>
#include <iomanip>
#include <vector>

//
// class declaration
//

class HBHEDarkeningAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HBHEDarkeningAnalyzer(const edm::ParameterSet&);
  ~HBHEDarkeningAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override {}
  void endJob() override {}
  void print(int ieta_min,
             int ieta_max,
             int lay_min,
             int lay_max,
             const HBHEDarkening* darkening,
             const HBHERecalibration& recalibration);

  // ----------member data ---------------------------
  double intlumi;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_topo_;
  edm::ESGetToken<HBHEDarkening, HBHEDarkeningRecord> tok_hbdark_;
  edm::ESGetToken<HBHEDarkening, HBHEDarkeningRecord> tok_hedark_;

  const HBHEDarkening* hb_darkening;
  const HBHEDarkening* he_darkening;
  HBHERecalibration hb_recalibration, he_recalibration;
  const HcalTopology* theTopology;
};

//
// constructors and destructor
//
HBHEDarkeningAnalyzer::HBHEDarkeningAnalyzer(const edm::ParameterSet& iConfig)
    : intlumi(iConfig.getParameter<double>("deliveredLumi")),
      hb_darkening(NULL),
      he_darkening(NULL),
      hb_recalibration(intlumi, 0, iConfig.getParameter<edm::FileInPath>("HBmeanenergies").fullPath()),
      he_recalibration(intlumi, 0, iConfig.getParameter<edm::FileInPath>("HEmeanenergies").fullPath()),
      theTopology(NULL) {
  tok_topo_ = esConsumes<HcalTopology, HcalRecNumberingRecord, edm::Transition::BeginRun>();
  tok_hbdark_ = esConsumes<HBHEDarkening, HBHEDarkeningRecord, edm::Transition::BeginRun>(edm::ESInputTag("", "HB"));
  tok_hedark_ = esConsumes<HBHEDarkening, HBHEDarkeningRecord, edm::Transition::BeginRun>(edm::ESInputTag("", "HE"));
}

HBHEDarkeningAnalyzer::~HBHEDarkeningAnalyzer() {}

//
// member functions
//

void HBHEDarkeningAnalyzer::beginJob() {}

void HBHEDarkeningAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  theTopology = &iSetup.getData(tok_topo_);
  hb_darkening = &iSetup.getData(tok_hbdark_);
  he_darkening = &iSetup.getData(tok_hedark_);

  //initialize recalibration classes
  std::vector<std::vector<int>> m_segmentation;
  int maxEta = theTopology->lastHBHERing();
  m_segmentation.resize(maxEta);
  for (int i = 0; i < maxEta; i++) {
    theTopology->getDepthSegmentation(i + 1, m_segmentation[i]);
  }
  std::cout << "HB: Eta " << theTopology->firstHBRing() << ":" << theTopology->lastHBRing() << " HE: Eta "
            << theTopology->firstHERing() << ":" << theTopology->lastHERing() << std::endl;
  hb_recalibration.setup(m_segmentation, hb_darkening);
  he_recalibration.setup(m_segmentation, he_darkening);
}

// ------------ method called on each new Event  ------------
void HBHEDarkeningAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::cout << std::setprecision(2);

  //HB tests
  int hb_ieta_min = theTopology->firstHBRing();
  int hb_ieta_max = theTopology->lastHBRing();
  int hb_lay_min = 1;
  int hb_lay_max = 17;
  std::cout << "HB" << std::endl;
  print(hb_ieta_min, hb_ieta_max, hb_lay_min, hb_lay_max, hb_darkening, hb_recalibration);

  std::cout << std::endl;

  //HE tests
  int he_ieta_min = theTopology->firstHERing();
  int he_ieta_max = theTopology->lastHERing();
  int he_lay_min = 1;
  int he_lay_max = 19;
  std::cout << "HE" << std::endl;
  print(he_ieta_min, he_ieta_max, he_lay_min, he_lay_max, he_darkening, he_recalibration);
}

void HBHEDarkeningAnalyzer::print(int ieta_min,
                                  int ieta_max,
                                  int lay_min,
                                  int lay_max,
                                  const HBHEDarkening* darkening,
                                  const HBHERecalibration& recalibration) {
  std::cout << "Darkening: ieta " << ieta_min << ":" << ieta_max << " layer " << lay_min << ":" << lay_max << std::endl;
  for (int ieta = ieta_min; ieta <= ieta_max; ++ieta) {
    std::cout << "Tower " << ieta << ": ";
    for (int lay = lay_min; lay <= lay_max; ++lay) {
      std::cout << darkening->degradation(intlumi, ieta, lay) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Recalibration: ieta " << ieta_min << ":" << ieta_max << " layer " << lay_min << ":" << lay_max
            << std::endl;
  for (int ieta = ieta_min; ieta <= ieta_max; ++ieta) {
    std::cout << "Tower " << ieta << ": ";
    for (int depth = 1; depth <= recalibration.maxDepth(); ++depth) {
      std::cout << recalibration.getCorr(ieta, depth) << " ";
    }
    std::cout << std::endl;
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HBHEDarkeningAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(HBHEDarkeningAnalyzer);
