#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/Span.h"

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCaloTower.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingFastJet.h"

#include "L1TriggerScouting/Utilities/interface/conversion.h"

// root libraries
#include "TLorentzVector.h"
#include "Math/VectorUtil.h"

// fastjet libraries
#include "fastjet/ClusterSequence.hh"

#include <memory>
#include <utility>
#include <vector>

using namespace l1ScoutingRun3;

class ScoutingJetProducer : public edm::stream::EDProducer<> {
public:
  explicit ScoutingJetProducer(const edm::ParameterSet&);
  ~ScoutingJetProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // tokens for scouting data
  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::CaloTower>> src_;
  double akR_;
  double ptMin_;
  bool debug_;
};

ScoutingJetProducer::ScoutingJetProducer(const edm::ParameterSet& iPSet)
    : src_(consumes(iPSet.getParameter<edm::InputTag>("src"))),
      akR_(iPSet.getParameter<double>("akR")),
      ptMin_(iPSet.getParameter<double>("ptMin")),
      debug_(iPSet.getParameter<bool>("debug")) {
  produces<OrbitCollection<l1ScoutingRun3::FastJet>>("FastJet").setBranchAlias("FastJetOrbitCollection");
}

// ------------ method called for each ORBIT  ------------
void ScoutingJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<OrbitCollection<l1ScoutingRun3::CaloTower>> caloTowerCollection;
  iEvent.getByToken(src_, caloTowerCollection);

  std::unique_ptr<OrbitCollection<l1ScoutingRun3::FastJet>> fastJetCollection(new FastJetOrbitCollection);
  std::vector<std::vector<l1ScoutingRun3::FastJet>> fastJetBuffer(3565);
  unsigned nFastJet = 0;

  // define fastjet algorithm
  fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, akR_);

  // loop over valid bunch crossings
  int nConst = 0;
  double area = 0.;
  for (const unsigned& bx : caloTowerCollection->getFilledBxs()) {
    const auto& cts = caloTowerCollection->bxIterator(bx);

    // create pseudojet vector to be filled
    std::vector<fastjet::PseudoJet> pjCTs;
    pjCTs.reserve(cts.size());

    // prepare pseudojets to give in input to fastjet
    for (const auto& ct : cts) {
      ROOT::Math::PtEtaPhiMVector ctLV(calol2::fEt(ct.hwEt()), calol2::fEta(ct.hwEta()), calol2::fPhi(ct.hwPhi()), 0);
      pjCTs.push_back(fastjet::PseudoJet(ctLV.px(), ctLV.py(), ctLV.pz(), ctLV.E()));
    }

    // run the jet clustering with the given jet definition
    fastjet::ClusterSequence clustSeq(pjCTs, jetDef);

    // get the resulting jets ordered in pt
    std::vector<fastjet::PseudoJet> incJets = fastjet::sorted_by_pt(clustSeq.inclusive_jets(ptMin_));

    // fill fast jet objects buffer
    fastJetBuffer[bx].reserve(incJets.size());
    for (const auto& incJet : incJets) {
      nConst = incJet.has_constituents()? incJet.constituents().size() : 0;
      area = incJet.has_area()? incJet.area() : -1.0;
      FastJet fj = FastJet(incJet.Et(), incJet.eta(), incJet.phi(), nConst, area);
      fastJetBuffer[bx].push_back(fj);
      nFastJet++;
    }
  }

  // fill orbit collection with reconstructed jets
  fastJetCollection->fillAndClear(fastJetBuffer, nFastJet);
  iEvent.put(std::move(fastJetCollection), "FastJet");
}

void ScoutingJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScoutingJetProducer);
