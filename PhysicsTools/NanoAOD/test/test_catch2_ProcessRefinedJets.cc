#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include <cmath>
#include <limits>

TEST_CASE("FastSim refinement support mask is shared with MET", "[ProcessRefinedJets]") {
  const bool guarded = GENERATE(false, true);
  std::string python = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("ProcessRefinedJets",
    jets=cms.InputTag("jets"), met=cms.InputTag("met"),
    refinedPtName=cms.string("ptrefined"), maskBtagName=cms.string("btag"),
    ptFinalName=cms.string("pt_final"), ptUnrefinedName=cms.string("pt_unrefined"))
)_";
  if (guarded) {
    python += "process.toTest.minGenJetPt = cms.double(10.)\n";
    python += "process.toTest.taggerNames = cms.vstring('score', 'ratio')\n";
    python +=
        "process.toTest.rawTaggerExpressions = cms.vstring(\"bDiscriminator('btag')\", "
        "\"?bDiscriminator('btag')>0?userFloat('rawRatio'):-1\")\n";
  }
  python += "process.moduleToTest(process.toTest)\n";
  edm::test::TestProcessor::Config config{python};
  auto jetToken = config.produces<std::vector<pat::Jet>>("jets");
  auto metToken = config.produces<std::vector<pat::MET>>("met");
  edm::test::TestProcessor tester(config);

  reco::GenJetCollection genJets(4);
  const std::vector<double> genPts{9.99, 10., 20., std::numeric_limits<double>::quiet_NaN()};
  for (unsigned int i = 0; i < genJets.size(); ++i)
    genJets[i].setP4(reco::Candidate::LorentzVector(genPts[i], 0., 0., genPts[i]));
  auto jets = std::make_unique<std::vector<pat::Jet>>(8);
  for (unsigned int i = 0; i < jets->size(); ++i) {
    auto &jet = jets->at(i);
    jet.setP4(reco::Candidate::LorentzVector(100., 0., 0., 100.));
    jet.addBDiscriminatorPair({"btag", i == 5 ? -1.f : (i == 7 ? 0.f : 0.5f)});
    jet.addUserFloat("ptrefined", 110.f);
    jet.addUserFloat("scorerefined", 0.75f);
    jet.addUserFloat("ratiorefined", 0.125f);
    jet.addUserFloat("rawRatio", 0.25f);
    if (guarded && i == 0) {
      // Invalid predictions on an unsupported jet must not affect the event.
      jet.addUserFloat("ptrefined", std::numeric_limits<float>::quiet_NaN(), true);
      jet.addUserFloat("scorerefined", std::numeric_limits<float>::infinity(), true);
    }
    if (i < genJets.size()) {
      reco::GenJetRef ref(&genJets, i);
      jet.setGenJetRef(edm::FwdRef<reco::GenJetCollection>(ref, ref));
    } else if (i == 6) {
      reco::GenJetRef unavailable(edm::ProductID(1, 1), static_cast<const reco::GenJet *>(nullptr), 0);
      jet.setGenJetRef(edm::FwdRef<reco::GenJetCollection>(unavailable, unavailable));
    }
  }
  auto met = std::make_unique<std::vector<pat::MET>>(1);
  met->front().setP4(reco::Candidate::LorentzVector(200., 0., 0., 200.));
  auto event = tester.test(std::make_pair(jetToken, std::move(jets)), std::make_pair(metToken, std::move(met)));
  auto result = event.get<std::vector<pat::Jet>>();
  REQUIRE(result->size() == 8);
  for (unsigned int i = 0; i < result->size(); ++i) {
    const bool refined = guarded ? (i == 1 || i == 2) : (i != 5 && i != 7);
    const auto &jet = result->at(i);
    REQUIRE(jet.userFloat("pt_final") == (refined ? 110.f : 100.f));
    REQUIRE(jet.userFloat("pt_unrefined") == 100.f);
    REQUIRE(jet.pt() == 100.);
    if (guarded) {
      REQUIRE(jet.userInt("fastSimRefinementApplied") == refined);
      REQUIRE(jet.userFloat("fastSimFinal_score") == (refined ? 0.75f : jet.bDiscriminator("btag")));
      REQUIRE(jet.userFloat("fastSimFinal_ratio") ==
              (refined ? 0.125f : (jet.bDiscriminator("btag") > 0 ? 0.25f : -1.f)));
    } else {
      REQUIRE_FALSE(jet.hasUserInt("fastSimRefinementApplied"));
      REQUIRE_FALSE(jet.hasUserFloat("fastSimFinal_score"));
    }
  }
  auto refinedMet = event.get<std::vector<pat::MET>>("Refined");
  REQUIRE(refinedMet->front().px() == (guarded ? 180. : 140.));
  REQUIRE(refinedMet->front().py() == 0.);
  REQUIRE(refinedMet->front().userFloat("pt_unrefined") == 200.f);
}

TEST_CASE("Invalid FastSim predictions fall back coherently without aborting the event", "[ProcessRefinedJets]") {
  const bool doMET = GENERATE(false, true);
  std::string python = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("ProcessRefinedJets",
    jets=cms.InputTag("jets"),
    refinedPtName=cms.string("ptrefined"), maskBtagName=cms.string("btag"),
    ptFinalName=cms.string("pt_final"), ptUnrefinedName=cms.string("pt_unrefined"),
    minGenJetPt=cms.double(10.), taggerNames=cms.vstring('score', 'ratio'),
    rawTaggerExpressions=cms.vstring("bDiscriminator('btag')", "userFloat('rawRatio')"))
)_";
  if (doMET)
    python += "process.toTest.met = cms.InputTag('met')\n";
  python += "process.moduleToTest(process.toTest)\n";
  edm::test::TestProcessor::Config config{python};
  auto jetToken = config.produces<std::vector<pat::Jet>>("jets");
  auto metToken = config.produces<std::vector<pat::MET>>("met");
  edm::test::TestProcessor tester(config);

  struct Prediction {
    float pt = 110.f;
    float score = 0.75f;
    float ratio = 0.125f;
    bool hasPt = true;
    bool hasScore = true;
    bool hasRatio = true;
    bool refine = false;
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<Prediction> predictions;
  for (float pt : {nan, inf, -inf, 0.f, -1.f})
    predictions.push_back({pt});
  for (float score : {nan, inf, -inf, -0.01f, 1.01f}) {
    predictions.push_back({110.f, score});
    predictions.push_back({110.f, 0.75f, score});
  }
  predictions.push_back({110.f, 0.75f, 0.125f, false});
  predictions.push_back({110.f, 0.75f, 0.125f, true, false});
  predictions.push_back({110.f, 0.75f, 0.125f, true, true, false});
  // Valid predictions, including exact score boundaries, remain unchanged.
  predictions.push_back({110.f, 0.f, 1.f, true, true, true, true});
  predictions.push_back({110.f, 1.f, 0.f, true, true, true, true});
  predictions.push_back({110.f, 0.75f, 0.125f, true, true, true, true});

  reco::GenJetCollection genJets(1);
  genJets.front().setP4(reco::Candidate::LorentzVector(20., 0., 0., 20.));
  reco::GenJetRef ref(&genJets, 0);
  auto jets = std::make_unique<std::vector<pat::Jet>>(predictions.size());
  double expectedPx = 200.;
  double expectedPy = 150.;
  for (unsigned int i = 0; i < predictions.size(); ++i) {
    const auto &prediction = predictions[i];
    auto &jet = jets->at(i);
    jet.setP4(i % 2 == 0 ? reco::Candidate::LorentzVector(100., 0., 0., 100.)
                         : reco::Candidate::LorentzVector(0., 100., 0., 100.));
    jet.setGenJetRef(edm::FwdRef<reco::GenJetCollection>(ref, ref));
    jet.addBDiscriminatorPair({"btag", 0.5f});
    jet.addUserFloat("rawRatio", -1.f);
    if (prediction.hasPt)
      jet.addUserFloat("ptrefined", prediction.pt);
    if (prediction.hasScore)
      jet.addUserFloat("scorerefined", prediction.score);
    if (prediction.hasRatio)
      jet.addUserFloat("ratiorefined", prediction.ratio);
    if (prediction.refine) {
      expectedPx += (100. - prediction.pt) * std::cos(jet.phi());
      expectedPy += (100. - prediction.pt) * std::sin(jet.phi());
    }
  }
  auto met = std::make_unique<std::vector<pat::MET>>(1);
  met->front().setP4(reco::Candidate::LorentzVector(200., 150., 0., 250.));
  auto event = tester.test(std::make_pair(jetToken, std::move(jets)), std::make_pair(metToken, std::move(met)));
  auto result = event.get<std::vector<pat::Jet>>();
  REQUIRE(result->size() == predictions.size());
  for (unsigned int i = 0; i < predictions.size(); ++i) {
    CAPTURE(i);
    const auto &prediction = predictions[i];
    const auto &jet = result->at(i);
    REQUIRE(jet.userInt("fastSimRefinementApplied") == prediction.refine);
    REQUIRE(jet.userFloat("pt_final") == (prediction.refine ? prediction.pt : 100.f));
    REQUIRE(jet.userFloat("pt_unrefined") == 100.f);
    REQUIRE(jet.pt() == 100.);
    REQUIRE(jet.userFloat("fastSimFinal_score") == (prediction.refine ? prediction.score : 0.5f));
    REQUIRE(jet.userFloat("fastSimFinal_ratio") == (prediction.refine ? prediction.ratio : -1.f));
  }
  if (doMET) {
    const auto refinedMet = event.get<std::vector<pat::MET>>("Refined");
    REQUIRE(refinedMet->size() == 1);
    REQUIRE(refinedMet->front().px() == expectedPx);
    REQUIRE(refinedMet->front().py() == expectedPy);
    REQUIRE(refinedMet->front().userFloat("pt_unrefined") == 250.f);
  }
}
