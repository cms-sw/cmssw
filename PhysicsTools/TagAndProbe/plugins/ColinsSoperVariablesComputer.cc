//

/**
  \class    ElectronConversionRejectionVars"
  \brief    Store electron partner track conversion-rejection quantities
            ("dist" and "dcot") in the TP tree.

  \author   Kalanand Mishra
  Fermi National Accelerator Laboratory
*/

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/TagAndProbe/interface/ColinsSoperVariables.h"
#include "TLorentzVector.h"

class ColinsSoperVariablesComputer : public edm::global::EDProducer<> {
public:
  explicit ColinsSoperVariablesComputer(const edm::ParameterSet& iConfig);
  ~ColinsSoperVariablesComputer() override;

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  const edm::EDGetTokenT<edm::View<reco::Candidate>> parentBosonToken_;
};

ColinsSoperVariablesComputer::ColinsSoperVariablesComputer(const edm::ParameterSet& iConfig)
    : parentBosonToken_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("parentBoson"))) {
  produces<edm::ValueMap<float>>("costheta");
  produces<edm::ValueMap<float>>("sin2theta");
  produces<edm::ValueMap<float>>("tanphi");
}

ColinsSoperVariablesComputer::~ColinsSoperVariablesComputer() {}

void ColinsSoperVariablesComputer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  // read input
  Handle<View<reco::Candidate>> bosons;
  iEvent.getByToken(parentBosonToken_, bosons);

  // prepare vector for output
  std::vector<float> values;
  std::vector<float> values2;
  std::vector<float> values3;

  // fill: use brute force
  double costheta = -10.0;
  double sin2theta = -10.0;
  double tanphi = -10.0;

  const reco::Candidate* daughter1 = nullptr;
  const reco::Candidate* daughter2 = nullptr;
  TLorentzVector mu(0., 0., 0., 0.);
  TLorentzVector mubar(0., 0., 0., 0.);
  bool isOS = false;
  int charge1 = 0, charge2 = 0;
  double res[3] = {-10., -10., -10.};

  View<reco::Candidate>::const_iterator boson, endbosons = bosons->end();

  for (boson = bosons->begin(); boson != endbosons; ++boson) {
    daughter1 = boson->daughter(0);
    daughter2 = boson->daughter(1);

    if (!(nullptr == daughter1 || nullptr == daughter2)) {
      charge1 = daughter1->charge();
      charge2 = daughter2->charge();
      isOS = charge1 * charge2 < 0;
      if (isOS && charge1 < 0) {
        mu.SetPxPyPzE(daughter1->px(), daughter1->py(), daughter1->pz(), daughter1->energy());
        mubar.SetPxPyPzE(daughter2->px(), daughter2->py(), daughter2->pz(), daughter2->energy());
      }
      if (isOS && charge1 > 0) {
        mu.SetPxPyPzE(daughter2->px(), daughter2->py(), daughter2->pz(), daughter2->energy());
        mubar.SetPxPyPzE(daughter1->px(), daughter1->py(), daughter1->pz(), daughter1->energy());
      }
    }

    calCSVariables(mu, mubar, res, boson->pz() < 0.0);

    costheta = res[0];
    sin2theta = res[1];
    tanphi = res[2];

    values.push_back(costheta);
    values2.push_back(sin2theta);
    values3.push_back(tanphi);
  }

  // convert into ValueMap and store
  auto valMap = std::make_unique<ValueMap<float>>();
  ValueMap<float>::Filler filler(*valMap);
  filler.insert(bosons, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap), "costheta");

  // ---> same for sin2theta
  auto valMap2 = std::make_unique<ValueMap<float>>();
  ValueMap<float>::Filler filler2(*valMap2);
  filler2.insert(bosons, values2.begin(), values2.end());
  filler2.fill();
  iEvent.put(std::move(valMap2), "sin2theta");

  // ---> same for tanphi
  auto valMap3 = std::make_unique<ValueMap<float>>();
  ValueMap<float>::Filler filler3(*valMap3);
  filler3.insert(bosons, values3.begin(), values3.end());
  filler3.fill();
  iEvent.put(std::move(valMap3), "tanphi");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ColinsSoperVariablesComputer);
