#include <memory>
#include <utility>
#include <vector>

#include "DataFormats/L1Scouting/interface/L1ScoutingCaloTower.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingFastJet.h"
#include "DataFormats/Math/interface/libminifloat.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"

class ScoutingJetProducer : public edm::global::EDProducer<> {
public:
  explicit ScoutingJetProducer(const edm::ParameterSet&);
  ~ScoutingJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // Number of BXs per orbit
  static constexpr unsigned int NBX = 3564;

  edm::EDGetTokenT<l1ScoutingRun3::CaloTowerOrbitCollection> const src_;
  double const akR_;
  double const ptMin_;
  int const towerMinHwEt_;
  int const towerMaxHwEt_;
  int const mantissaPrecision_;
};

ScoutingJetProducer::ScoutingJetProducer(const edm::ParameterSet& iPSet)
    : src_(consumes(iPSet.getParameter<edm::InputTag>("src"))),
      akR_(iPSet.getParameter<double>("akR")),
      ptMin_(iPSet.getParameter<double>("ptMin")),
      towerMinHwEt_(iPSet.getParameter<int>("towerMinHwEt")),
      towerMaxHwEt_(iPSet.getParameter<int>("towerMaxHwEt")),
      mantissaPrecision_(iPSet.getParameter<int>("mantissaPrecision")) {
  produces<l1ScoutingRun3::FastJetOrbitCollection>("FastJet").setBranchAlias("FastJetOrbitCollection");
}

// ------------ method called for each ORBIT  ------------
void ScoutingJetProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  auto const& caloTowerCollection = iEvent.get(src_);

  auto fastJetCollection = std::make_unique<l1ScoutingRun3::FastJetOrbitCollection>();
  std::vector<std::vector<l1ScoutingRun3::FastJet>> fastJetBuffer(NBX + 1);
  unsigned int nFastJet = 0;

  // define fastjet algorithm
  fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, akR_);

  // create pseudojet vector to be filled
  std::vector<fastjet::PseudoJet> pjCTs;

  // loop over valid bunch crossings
  for (auto const bx : caloTowerCollection.getFilledBxs()) {
    LogTrace("ScoutingJetProducer") << "[ScoutingJetProducer:" << moduleDescription().moduleLabel() << "] BX = " << bx;
    LogTrace("ScoutingJetProducer") << "[ScoutingJetProducer:" << moduleDescription().moduleLabel()
                                    << "]   Inputs (l1ScoutingRun3::CaloTower and fastjet::PseudoJet)";

    auto const& cts = caloTowerCollection.bxIterator(bx);

    // prepare PseudoJets to give in input to fastjet
    pjCTs.clear();
    pjCTs.reserve(cts.size());
    for (auto const& ct : cts) {
      if (not((towerMinHwEt_ < 0 or ct.hwEt() >= towerMinHwEt_) and
              (towerMaxHwEt_ < 0 or ct.hwEt() <= towerMaxHwEt_))) {
        continue;
      }

      if (not l1ScoutingRun3::calol1::validHwEta(ct.hwEta())) {
        edm::LogWarning("ScoutingJetProducer") << "CaloTower in BX=" << bx << " with invalid hwEta value ("
                                               << ct.hwEta() << ") will not be used for jet clustering !";
        continue;
      }

      if (not l1ScoutingRun3::calol1::validHwPhi(ct.hwPhi())) {
        edm::LogWarning("ScoutingJetProducer") << "CaloTower in BX=" << bx << " with invalid hwPhi value ("
                                               << ct.hwPhi() << ") will not be used for jet clustering !";
        continue;
      }

      float const ctEt = l1ScoutingRun3::calol1::fEt(ct.hwEt());
      float const ctEta = l1ScoutingRun3::calol1::fEta(ct.hwEta());
      float const ctPhi = l1ScoutingRun3::calol1::fPhi(ct.hwPhi());

      pjCTs.emplace_back(fastjet::PtYPhiM(ctEt, ctEta, ctPhi, 0));

      LogTrace("ScoutingJetProducer") << "[ScoutingJetProducer:" << moduleDescription().moduleLabel() << "]     ["
                                      << (pjCTs.size() - 1) << "] hwEt=" << ct.hwEt() << " hwEta=" << ct.hwEta()
                                      << " hwPhi=" << ct.hwPhi() << " (PseudoJet: pt=" << ctEt << " eta=" << ctEta
                                      << " phi=" << ctPhi << " px=" << pjCTs.back().px() << " py=" << pjCTs.back().py()
                                      << " pz=" << pjCTs.back().pz() << " E=" << pjCTs.back().E() << ")";
    }

    // run the jet clustering with the given jet definition
    fastjet::ClusterSequence clustSeq(pjCTs, jetDef);

    // get the resulting jets ordered in pt
    std::vector<fastjet::PseudoJet> incJets = fastjet::sorted_by_pt(clustSeq.inclusive_jets(ptMin_));

    LogTrace("ScoutingJetProducer") << "[ScoutingJetProducer:" << moduleDescription().moduleLabel()
                                    << "]   Outputs (l1ScoutingRun3::FastJet)";

    // fill l1ScoutingRun3::FastJet objects buffer
    auto& bufferThisBX = fastJetBuffer[bx];
    bufferThisBX.reserve(incJets.size());
    for (auto const& incJet : incJets) {
      int const nConst = incJet.has_constituents() ? incJet.constituents().size() : 0;
      float const area = incJet.has_area() ? incJet.area() : -1.0f;
      bufferThisBX.emplace_back(MiniFloatConverter::reduceMantissaToNbitsRounding(incJet.Et(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(incJet.eta(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(incJet.phi_std(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(incJet.m(), mantissaPrecision_),
                                nConst,
                                MiniFloatConverter::reduceMantissaToNbitsRounding(area, mantissaPrecision_));
      ++nFastJet;

      LogTrace("ScoutingJetProducer") << "[ScoutingJetProducer:" << moduleDescription().moduleLabel() << "]     ["
                                      << (bufferThisBX.size() - 1) << "] et=" << bufferThisBX.back().et()
                                      << " eta=" << bufferThisBX.back().eta() << " phi=" << bufferThisBX.back().phi()
                                      << " mass=" << bufferThisBX.back().mass()
                                      << " nConst=" << bufferThisBX.back().nConst()
                                      << " area=" << bufferThisBX.back().area();
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
  desc.add<int>("towerMinHwEt", 1)
      ->setComment("Min hwEt (inclusive) of CaloTowers used for jet clustering (ignored if negative)");
  desc.add<int>("towerMaxHwEt", -1)
      ->setComment("Max hwEt (inclusive) of CaloTowers used for jet clustering (ignored if negative)");
  desc.add<int>("mantissaPrecision", 10)->setComment("default float16, change to 23 for float32");
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ScoutingJetProducer);
