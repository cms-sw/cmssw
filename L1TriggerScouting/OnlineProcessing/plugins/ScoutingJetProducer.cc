#include <memory>
#include <utility>
#include <vector>

#include "DataFormats/L1Scouting/interface/L1ScoutingCaloTower.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingFastJet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

// ROOT libraries
#include "Math/Vector4D.h"
#include "Math/VectorUtil.h"

// fastjet libraries
#include "fastjet/ClusterSequence.hh"

class ScoutingJetProducer : public edm::global::EDProducer<> {
public:
  explicit ScoutingJetProducer(const edm::ParameterSet&);
  ~ScoutingJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<l1ScoutingRun3::CaloTowerOrbitCollection> const src_;
  double const akR_;
  double const ptMin_;
  bool const debug_;
};

ScoutingJetProducer::ScoutingJetProducer(const edm::ParameterSet& iPSet)
    : src_(consumes(iPSet.getParameter<edm::InputTag>("src"))),
      akR_(iPSet.getParameter<double>("akR")),
      ptMin_(iPSet.getParameter<double>("ptMin")),
      debug_(iPSet.getUntrackedParameter<bool>("debug")) {
  produces<l1ScoutingRun3::FastJetOrbitCollection>("FastJet").setBranchAlias("FastJetOrbitCollection");
}

// ------------ method called for each ORBIT  ------------
void ScoutingJetProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  auto const& caloTowerCollection = iEvent.get(src_);

  auto fastJetCollection = std::make_unique<l1ScoutingRun3::FastJetOrbitCollection>();
  std::vector<std::vector<l1ScoutingRun3::FastJet>> fastJetBuffer(3565);  // range of BX values
  unsigned int nFastJet = 0;

  // define fastjet algorithm
  fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, akR_);

  // create pseudojet vector to be filled
  std::vector<fastjet::PseudoJet> pjCTs;

  // loop over valid bunch crossings
  for (auto const bx : caloTowerCollection.getFilledBxs()) {
    const auto& cts = caloTowerCollection.bxIterator(bx);

    // prepare pseudojets to give in input to fastjet
    pjCTs.clear();
    pjCTs.reserve(cts.size());
    for (const auto& ct : cts) {
      ROOT::Math::PtEtaPhiMVector ctLV(l1ScoutingRun3::calol1::fEt(ct.hwEt()),
                                       l1ScoutingRun3::calol1::fEta(ct.hwEta()),
                                       l1ScoutingRun3::calol1::fPhi(ct.hwPhi()),
                                       0);
      pjCTs.emplace_back(ctLV.px(), ctLV.py(), ctLV.pz(), ctLV.E());
    }

    // run the jet clustering with the given jet definition
    fastjet::ClusterSequence clustSeq(pjCTs, jetDef);

    // get the resulting jets ordered in pt
    std::vector<fastjet::PseudoJet> incJets = fastjet::sorted_by_pt(clustSeq.inclusive_jets(ptMin_));

    // fill fast jet objects buffer
    auto& bufferThisBX = fastJetBuffer[bx];
    bufferThisBX.reserve(incJets.size());
    for (const auto& incJet : incJets) {
      int nConst = incJet.has_constituents() ? incJet.constituents().size() : 0;
      float area = incJet.has_area() ? incJet.area() : -1.0f;
      bufferThisBX.emplace_back(incJet.Et(), incJet.eta(), incJet.phi(), incJet.m(), nConst, area);
      nFastJet++;
    }
  }

  // fill orbit collection with reconstructed jets
  fastJetCollection->fillAndClear(fastJetBuffer, nFastJet);
  iEvent.put(std::move(fastJetCollection), "FastJet");
}

void ScoutingJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  desc.add<double>("akR");
  desc.add<double>("ptMin");
  desc.addUntracked<bool>("debug", false);
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ScoutingJetProducer);
