#include "catch2/catch_all.hpp"
// ROOT's Math/StaticCheck.h (pulled in via reco::LeafCandidate) defines its own STATIC_CHECK
#undef STATIC_CHECK

#include "HLTrigger/HLTcore/interface/getL1TCandidates.h"

namespace {
  using LorentzVector = reco::LeafCandidate::LorentzVector;
  using PolarLorentzVector = reco::LeafCandidate::PolarLorentzVector;

  LorentzVector makeP4(double pt, double eta, double phi) {
    return LorentzVector(PolarLorentzVector(pt, eta, phi, 0.));
  }

  l1t::JetBxCollection makeRun3Jets() {
    l1t::JetBxCollection jets(0, 0, 0);
    jets.push_back(0, l1t::Jet(makeP4(100., 2.5, 0.5)));
    jets.push_back(0, l1t::Jet(makeP4(50., -3.0, -1.0)));
    return jets;
  }

  l1t::TauBxCollection makeRun3Taus() {
    l1t::TauBxCollection taus(0, 0, 0);
    taus.push_back(0, l1t::Tau(makeP4(40., 1.0, -0.5)));
    return taus;
  }

  l1t::EGammaBxCollection makeRun3EGammas() {
    l1t::EGammaBxCollection egs(0, 0, 0);
    egs.push_back(0, l1t::EGamma(makeP4(26., -1.2, 0.3)));
    return egs;
  }

  l1t::P2GTCandidateCollection makePhase2Cands() {
    l1t::P2GTCandidateCollection cands(4);
    cands[0].setP4(makeP4(120., 2.0, 1.0));
    cands[1].setP4(makeP4(60., -2.0, -2.0));
    cands[2].setP4(makeP4(30., 0.1, 3.0));
    cands[3].setP4(makeP4(45., -0.7, 2.2));
    return cands;
  }

  void requireSameP4(LorentzVector const& a, LorentzVector const& b) {
    REQUIRE(a.pt() == Catch::Approx(b.pt()));
    REQUIRE(a.eta() == Catch::Approx(b.eta()));
    REQUIRE(a.phi() == Catch::Approx(b.phi()));
  }
}  // namespace

TEST_CASE("hlt::getL1TCandidates", "[getL1TCandidates]") {
  auto const run3Jets = makeRun3Jets();
  auto const run3Taus = makeRun3Taus();
  auto const run3EGammas = makeRun3EGammas();
  auto const phase2Cands = makePhase2Cands();

  SECTION("empty") {
    trigger::TriggerRefsCollections refs;
    REQUIRE(hlt::getL1TJets(refs).empty());
    REQUIRE(hlt::getL1TTaus(refs).empty());
    REQUIRE(hlt::getL1TEGammas(refs).empty());
  }

  SECTION("Run-3 l1t::Jet refs") {
    trigger::TriggerRefsCollections refs;
    for (unsigned int i = 0; i < run3Jets.size(); ++i)
      refs.addObject(trigger::TriggerL1Jet, l1t::JetRef(&run3Jets, i));

    auto const cands = hlt::getL1TJets(refs);
    REQUIRE(cands.size() == 2);
    requireSameP4(cands[0]->p4(), run3Jets[0].p4());
    requireSameP4(cands[1]->p4(), run3Jets[1].p4());
  }

  SECTION("Phase-2 l1t::P2GTCandidate refs as jets") {
    trigger::TriggerRefsCollections refs;
    for (unsigned int i = 0; i < 3; ++i)
      refs.addObject(trigger::TriggerL1Jet, l1t::P2GTCandidateRef(&phase2Cands, i));

    auto const cands = hlt::getL1TJets(refs);
    REQUIRE(cands.size() == 3);
    for (unsigned int i = 0; i < 3; ++i)
      requireSameP4(cands[i]->p4(), phase2Cands[i].p4());
  }

  SECTION("jets: both kinds, and other trigger types are ignored") {
    trigger::TriggerRefsCollections refs;
    refs.addObject(trigger::TriggerL1Jet, l1t::P2GTCandidateRef(&phase2Cands, 0));
    refs.addObject(trigger::TriggerL1Tau, l1t::P2GTCandidateRef(&phase2Cands, 1));
    refs.addObject(trigger::TriggerL1Jet, l1t::JetRef(&run3Jets, 1));

    auto const cands = hlt::getL1TJets(refs);
    REQUIRE(cands.size() == 2);
    requireSameP4(cands[0]->p4(), run3Jets[1].p4());
    requireSameP4(cands[1]->p4(), phase2Cands[0].p4());
  }

  SECTION("taus: Run-3 l1t::Tau and Phase-2 CL2Taus (TriggerL1Tau)") {
    trigger::TriggerRefsCollections refs;
    refs.addObject(trigger::TriggerL1Tau, l1t::P2GTCandidateRef(&phase2Cands, 1));
    refs.addObject(trigger::TriggerL1Tau, l1t::TauRef(&run3Taus, 0));
    refs.addObject(trigger::TriggerL1Jet, l1t::JetRef(&run3Jets, 0));

    auto const cands = hlt::getL1TTaus(refs);
    REQUIRE(cands.size() == 2);
    requireSameP4(cands[0]->p4(), run3Taus[0].p4());
    requireSameP4(cands[1]->p4(), phase2Cands[1].p4());
  }

  SECTION("e/gammas: Run-3 l1t::EGamma and Phase-2 CL2Electrons (TriggerL1EG), seed jets ignored") {
    trigger::TriggerRefsCollections refs;
    refs.addObject(trigger::TriggerL1Jet, l1t::P2GTCandidateRef(&phase2Cands, 0));
    refs.addObject(trigger::TriggerL1EG, l1t::P2GTCandidateRef(&phase2Cands, 2));
    refs.addObject(trigger::TriggerL1Jet, l1t::P2GTCandidateRef(&phase2Cands, 1));
    refs.addObject(trigger::TriggerL1EG, l1t::EGammaRef(&run3EGammas, 0));

    auto const cands = hlt::getL1TEGammas(refs);
    REQUIRE(cands.size() == 2);
    requireSameP4(cands[0]->p4(), run3EGammas[0].p4());
    requireSameP4(cands[1]->p4(), phase2Cands[2].p4());
  }

  SECTION("e/gammas: Phase-2 GCT e/gammas (TriggerL1IsoEG, TriggerL1NoIsoEG) are read as P2GT refs only") {
    trigger::TriggerRefsCollections refs;
    refs.addObject(trigger::TriggerL1NoIsoEG, l1t::P2GTCandidateRef(&phase2Cands, 1));
    refs.addObject(trigger::TriggerL1IsoEG, l1t::P2GTCandidateRef(&phase2Cands, 0));
    refs.addObject(trigger::TriggerL1EG, l1t::P2GTCandidateRef(&phase2Cands, 2));
    refs.addObject(trigger::TriggerL1IsoEG, l1t::EGammaRef(&run3EGammas, 0));

    auto const cands = hlt::getL1TEGammas(refs);
    REQUIRE(cands.size() == 3);
    requireSameP4(cands[0]->p4(), phase2Cands[2].p4());
    requireSameP4(cands[1]->p4(), phase2Cands[0].p4());
    requireSameP4(cands[2]->p4(), phase2Cands[1].p4());
  }

  SECTION("Phase-2 GCT/GTT taus (TriggerL1TauJet): P2GT refs only, no Run-3 ref type") {
    trigger::TriggerRefsCollections refs;
    refs.addObject(trigger::TriggerL1TauJet, l1t::P2GTCandidateRef(&phase2Cands, 3));
    refs.addObject(trigger::TriggerL1Tau, l1t::P2GTCandidateRef(&phase2Cands, 2));

    auto const cands = hlt::getL1TCandidates<>(refs, trigger::TriggerL1TauJet);
    REQUIRE(cands.size() == 1);
    requireSameP4(cands[0]->p4(), phase2Cands[3].p4());
  }
}
