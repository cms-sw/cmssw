// PATGenJetSubstructurePacker with TestProcessor and hand-made inputs.
// Chain: inputs -> PATGenJetProducer (fat jets) and PATGenJetProducer (subjets) -> packer.
// Checks: first-within-distMax match; subjet output order = groomed daughter order; fat-jet subjet keys
// point into the packer's own SubJets output; ECF copied with stripped label; groomedMass = mass of the
// subjet sum; genJetFwdRef and genCandidatesFwdPtr of fat jets and subjets kept; no tagInfos product;
// association; fixDaughters.

#include "DataFormats/PatCandidates/interface/GenJet.h"
#include "PhysicsTools/PatAlgos/test/PatGenJetTestCommon.h"

#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"

#include <cmath>
#include <string>
#include <vector>

// DataFormats/Math/interface/Error.h defines a STATIC_CHECK macro that clashes with Catch2
#undef STATIC_CHECK
#include <catch2/catch_all.hpp>

using Catch::Approx;

namespace {
  const std::string kInputs = "genInputsForPatGenJets";

  // extra: more arguments of the packer; fatArgs: more arguments of the fat-jet producer
  std::string packerConfig(std::string const& extra = "", std::string const& fatArgs = "") {
    return patgenjettest::header() +
           "process.patGenJets = cms.EDProducer('PATGenJetProducer', jetSource = cms.InputTag('" + kInputs +
           ":jets'), packedGenParticles = cms.InputTag('" + kInputs + "')" + fatArgs + ")\n" +
           "process.patSubjets = cms.EDProducer('PATGenJetProducer', jetSource = cms.InputTag('" + kInputs +
           ":subjets'), packedGenParticles = cms.InputTag('" + kInputs + "'))\n" +
           "process.toTest = cms.EDProducer('PATGenJetSubstructurePacker', jetSrc = cms.InputTag('patGenJets'),\n"
           "  distMax = cms.double(0.8), labelPostfixesToStrip = cms.vstring('ForPatGenJets'),\n"
           "  algos = cms.VPSet(cms.PSet(label = cms.string('SoftDropPuppi'), groomedJets = cms.InputTag('" +
           kInputs + ":groomed'),\n    subjets = cms.InputTag('patSubjets'), groomedUserFloats = cms.VInputTag('" +
           kInputs + ":ecfN2'), subjetInstance = cms.string('SubJets')))" + extra + ")\n" +
           "process.moduleToTest(process.toTest, cms.Task(process." + kInputs +
           ", process.patGenJets, process.patSubjets))\n";
  }

  std::vector<unsigned> subjetKeys(pat::GenJet const& j) {
    std::vector<unsigned> r;
    for (auto const& p : j.subjets("SoftDropPuppi"))
      r.push_back(p.key());
    return r;
  }
}  // namespace

TEST_CASE("PATGenJetSubstructurePacker: subjet output, matching, ECF, groomedMass", "[PATGenJetSubstructurePacker]") {
  edm::test::TestProcessor tester(packerConfig());
  auto event = tester.test();
  auto const& fat = *event.get<std::vector<pat::GenJet>>();
  auto const& sub = *event.get<std::vector<pat::GenJet>>("SubJets");

  // subjet output in groomed-daughter order: G0 = {S2, S0}, G1 = {S1, S3}, G2 = {S4, S5};
  // the subjet producer sorted them by pt (50, 30, 25, 20, 10, 5), so this order is the packer's
  REQUIRE(sub.size() == 6);
  CHECK_THAT(patgenjettest::pts(sub), Catch::Matchers::Approx(std::vector<double>{30., 50., 20., 25., 10., 5.}));
  CHECK(patgenjettest::badBackRefs(sub) == 0);

  // fat jets: producer order (pt-sorted) 80, 50, 30, 12; genJetFwdRef kept
  REQUIRE(fat.size() == 4);
  CHECK_THAT(patgenjettest::pts(fat), Catch::Matchers::Approx(std::vector<double>{80., 50., 30., 12.}));
  CHECK(patgenjettest::badBackRefs(fat) == 0);

  // "80" matches G0 (dR ~0.1): subjets = output entries 0, 1
  auto const& j80 = fat[0];
  REQUIRE(j80.hasSubjets("SoftDropPuppi"));
  CHECK(subjetKeys(j80) == std::vector<unsigned>{0, 1});
  // the Ptrs point into the packer's OWN SubJets product: they resolve to its entries
  for (auto const& p : j80.subjets("SoftDropPuppi")) {
    CHECK(p.get() == &sub[p.key()]);
    CHECK(p.id() != sub[0].genJetFwdRef().backRef().id());  // the packer output, not the input subjets
  }
  CHECK(j80.userFloat("genInputs:ecfN2") == Approx(0.11f));  // label stripped of "ForPatGenJets"
  CHECK_FALSE(j80.hasUserFloat("genInputsForPatGenJets:ecfN2"));
  const double m = (sub[0].p4() + sub[1].p4()).mass();
  CHECK(j80.groomedMass("SoftDropPuppi") == Approx(m));
  CHECK(j80.groomedMass(0) == Approx(m));
  CHECK(m > 0);

  // "50": G1 at dR ~0.5 comes before G2 at dR ~0.05: first within distMax wins, not the closest
  auto const& j50 = fat[1];
  CHECK(subjetKeys(j50) == std::vector<unsigned>{2, 3});
  CHECK(j50.userFloat("genInputs:ecfN2") == Approx(0.22f));

  // "30", "12": no groomed jet within distMax: empty collection under the label, no ECF, groomedMass -1
  for (size_t i : {2u, 3u}) {
    CHECK(fat[i].hasSubjets("SoftDropPuppi"));
    CHECK(fat[i].subjets("SoftDropPuppi").empty());
    CHECK(fat[i].groomedMass("SoftDropPuppi") == -1.0);
    CHECK_FALSE(fat[i].hasUserFloat("genInputs:ecfN2"));
  }

  // no tagInfos product
  CHECK_THROWS_AS(event.get<edm::OwnVector<reco::BaseTagInfo>>("tagInfos")->size(), cms::Exception);

  // fat jets keep their packed constituents (fixDaughters False)
  CHECK(j80.numberOfDaughters() == 3);
  // subjets keep their re-keyed constituents and their Specific/flavour defaults
  CHECK(sub[0].numberOfDaughters() == 1);

  // association old -> new: identity
  auto const& assoc = *event.get<edm::Association<std::vector<pat::GenJet>>>();
  REQUIRE(assoc.ids().size() == 1);
  for (unsigned i = 0; i < 4; ++i)
    CHECK(assoc.get(assoc.ids()[0].first, i).get() == &fat[i]);
}

TEST_CASE("PATGenJetSubstructurePacker: distMax", "[PATGenJetSubstructurePacker]") {
  // distMax 0.3: "50" now skips G1 (dR ~0.5) and takes G2
  edm::test::TestProcessor tester(packerConfig() + "process.toTest.distMax = 0.3\n");
  auto event = tester.test();
  auto const& fat = *event.get<std::vector<pat::GenJet>>();
  REQUIRE(fat.size() == 4);
  CHECK(subjetKeys(fat[1]) == std::vector<unsigned>{4, 5});
  CHECK(fat[1].userFloat("genInputs:ecfN2") == Approx(0.33f));
}

TEST_CASE("PATGenJetSubstructurePacker: fixDaughters", "[PATGenJetSubstructurePacker]") {
  edm::test::TestProcessor tester(packerConfig(", fixDaughters = cms.bool(True)"));
  auto event = tester.test();
  auto const& fat = *event.get<std::vector<pat::GenJet>>();
  auto const& sub = *event.get<std::vector<pat::GenJet>>("SubJets");
  REQUIRE(fat.size() == 4);
  auto const& j80 = fat[0];
  // stored daughters: the two subjets, then the constituent not in a subjet (packed key 0)
  REQUIRE(j80.reco::CompositePtrCandidate::numberOfDaughters() == 3);
  CHECK(j80.reco::CompositePtrCandidate::daughterPtr(0).get() == &sub[0]);
  CHECK(j80.reco::CompositePtrCandidate::daughterPtr(1).get() == &sub[1]);
  CHECK(j80.reco::CompositePtrCandidate::daughterPtr(2).key() == 0);
  // pat::GenJet flattens subjets back to constituents (Jet.cc:616 copy): 3 constituents
  CHECK(j80.numberOfDaughters() == 3);
  // a fat jet without subjets keeps its constituents
  CHECK(fat[2].reco::CompositePtrCandidate::numberOfDaughters() == 2);
}

TEST_CASE("PATGenJetSubstructurePacker: duplicate subjet instance rejected", "[PATGenJetSubstructurePacker]") {
  const std::string extra = "\nprocess.toTest.algos.append(process.toTest.algos[0].clone(label = 'Other'))\n";
  CHECK_THROWS_AS(edm::test::TestProcessor(packerConfig() + extra), cms::Exception);
}

TEST_CASE("PATGenJetSubstructurePacker: genJetFwdRef and genCandidatesFwdPtr unchanged",
          "[PATGenJetSubstructurePacker]") {
  const std::string fatArgs = ", genJetForwardMap = cms.InputTag('" + kInputs +
                              ":jetmap'), embedGenCandidates = cms.bool(True), genCandidatesForwardMaps = "
                              "cms.VInputTag('" +
                              kInputs + ":map1', '" + kInputs + ":map2')";
  edm::test::TestProcessor tester(packerConfig("", fatArgs));
  auto event = tester.test();
  auto const& fat = *event.get<std::vector<pat::GenJet>>();
  auto const& sub = *event.get<std::vector<pat::GenJet>>("SubJets");
  REQUIRE(fat.size() == 4);  // 80, 50, 30, 12
  CHECK(patgenjettest::badBackRefs(fat) == 0);
  CHECK(patgenjettest::badBackRefs(sub) == 0);
  const std::vector<std::vector<int>> expectedStage2 = {{0, 1, 5, 9}, {4, 3}, {2, 6}, {7, 8}};
  for (size_t i = 0; i < fat.size(); ++i) {
    INFO("jet pt " << fat[i].pt());
    auto const& l = fat[i].genJetFwdRef();
    CHECK(l.ref().key() == static_cast<unsigned>(patgenjettest::expectedJetFwdKey(fat[i].pt())));
    auto const& fp = fat[i].genCandidatesFwdPtr();
    REQUIRE(fp.size() == expectedStage2[i].size());
    for (size_t k = 0; k < fp.size(); ++k) {
      CHECK(fp[k].ptr().key() == static_cast<unsigned>(expectedStage2[i][k]));
    }
  }
  // subjets: made without maps and without embedGenCandidates
  for (auto const& j : sub)
    CHECK(j.genCandidatesFwdPtr().empty());
}
