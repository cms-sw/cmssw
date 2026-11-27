// PATGenJetProducer with TestProcessor and hand-made inputs.
// Checks: re-keying (both daughterOrder values), sorting, genJetFwdRef after the sort (with and without
// genJetForwardMap), embedGenCandidates (with and without genCandidatesForwardMaps), user data with
// stripped postfix.

#include "DataFormats/PatCandidates/interface/GenJet.h"
#include "PhysicsTools/PatAlgos/test/PatGenJetTestCommon.h"

#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cmath>
#include <string>
#include <vector>

// DataFormats/Math/interface/Error.h defines a STATIC_CHECK macro that clashes with Catch2
#undef STATIC_CHECK
#include <catch2/catch_all.hpp>

using Catch::Approx;
using patgenjettest::header;

namespace {
  const std::string kInputs = "genInputsForPatGenJets";

  // extra: more arguments of the module; post: python lines run after it (to override a base argument)
  std::string producerConfig(std::string const& extra,
                             std::string const& post = "",
                             bool unmapped = false,
                             bool drop = false) {
    return header(false, unmapped, drop) +
           "process.toTest = cms.EDProducer('PATGenJetProducer',\n  jetSource = cms.InputTag('" + kInputs +
           ":jets'),\n  packedGenParticles = cms.InputTag('" + kInputs + "')" + extra + ")\n" + post +
           "process.moduleToTest(process.toTest, cms.Task(process." + kInputs + "))\n";
  }

  std::vector<unsigned> keys(pat::GenJet const& j) {
    std::vector<unsigned> r;
    for (size_t i = 0; i < j.numberOfDaughters(); ++i)
      r.push_back(j.daughterPtr(i).key());
    return r;
  }

  pat::GenJet const& jetWithPt(std::vector<pat::GenJet> const& v, double pt) {
    for (auto const& j : v)
      if (std::abs(j.pt() - pt) < 1e-6)
        return j;
    throw std::runtime_error("no jet with this pt");
  }
}  // namespace

TEST_CASE("PATGenJetProducer: sorting, back refs, re-keying in pt order, Specific", "[PATGenJetProducer]") {
  edm::test::TestProcessor tester(producerConfig(""));
  auto event = tester.test();
  auto const& out = *event.get<std::vector<pat::GenJet>>();

  // sorting: input order is 30, 80, 50, 12
  REQUIRE(out.size() == 4);
  CHECK_THAT(patgenjettest::pts(out), Catch::Matchers::Approx(std::vector<double>{80., 50., 30., 12.}));
  // the original object is the input jet (input index of the "80" jet is 1)
  CHECK(out[0].originalObjectRef().key() == 1);

  // genJetFwdRef after the sort: back ref = the input jet; without genJetForwardMap the forward ref is the same
  CHECK(patgenjettest::badBackRefs(out) == 0);
  for (auto const& j : out) {
    CHECK(j.genJetFwdRef().ref() == j.genJetFwdRef().backRef());
    CHECK(j.genJet() == &j);
  }

  // re-keying, daughterOrder "pt" (default): gens {1, 5, 8, 9} -> packed {7, 3, 0, 0}; duplicate dropped
  auto const& j80 = out[0];
  CHECK(keys(j80) == std::vector<unsigned>{7, 3, 0});
  const edm::ProductID origId = j80.originalObject()->sourceCandidatePtr(0).id();  // the genParticles product
  for (size_t i = 0; i < j80.numberOfDaughters(); ++i) {
    CHECK(j80.daughterPtr(i).id() != origId);
    CHECK(j80.daughterPtr(i).id() == j80.daughterPtr(0).id());
    CHECK(dynamic_cast<pat::PackedGenParticle const*>(j80.daughterPtr(i).get()) != nullptr);
  }
  // constituent kinematics preserved (packing precision); gen pt of gens 1, 5, 8 = 40, 30, 22
  CHECK(j80.daughterPtr(0)->pt() == Approx(40.).epsilon(0.01));
  CHECK(j80.daughterPtr(1)->pt() == Approx(30.).epsilon(0.01));
  CHECK(j80.daughterPtr(2)->pt() == Approx(22.).epsilon(0.01));
  CHECK(keys(jetWithPt(out, 50.)) == std::vector<unsigned>{1, 8});
  CHECK(keys(jetWithPt(out, 12.)) == std::vector<unsigned>{4, 2});

  // Specific kept (emEnergy = pt/4, hadEnergy = pt/2 in the inputs)
  CHECK(j80.emEnergy() == Approx(20.));
  CHECK(j80.hadEnergy() == Approx(40.));
  CHECK(j80.jetArea() == Approx(0.5));

  // the tagInfos product is gone
  CHECK_THROWS_AS(event.get<edm::OwnVector<reco::BaseTagInfo>>("tagInfos")->size(), cms::Exception);
}

TEST_CASE("PATGenJetProducer: daughterOrder key reproduces PATGenJetSlimmer order", "[PATGenJetProducer]") {
  edm::test::TestProcessor tester(producerConfig(",\n  daughterOrder = cms.string('key')"));
  auto event = tester.test();
  auto const& out = *event.get<std::vector<pat::GenJet>>();
  REQUIRE(out.size() == 4);
  CHECK(keys(out[0]) == std::vector<unsigned>{0, 3, 7});
  CHECK(keys(jetWithPt(out, 12.)) == std::vector<unsigned>{2, 4});
  CHECK(patgenjettest::badBackRefs(out) == 0);
}

TEST_CASE("PATGenJetProducer: empty packedGenParticles keeps the original constituents", "[PATGenJetProducer]") {
  edm::test::TestProcessor tester(producerConfig("", "process.toTest.packedGenParticles = ''\n"));
  auto event = tester.test();
  auto const& out = *event.get<std::vector<pat::GenJet>>();
  REQUIRE(out.size() == 4);
  CHECK(keys(out[0]) == std::vector<unsigned>{1, 5, 8, 9});
  CHECK(out[0].daughterPtr(0).id() == out[0].originalObject()->sourceCandidatePtr(0).id());
  CHECK(out[0].getGenConstituents().size() == 4);  // GenParticle constituents: works
}

TEST_CASE("PATGenJetProducer: a constituent without packed partner is dropped", "[PATGenJetProducer]") {
  // the input jet "30" has a third daughter (gen 10) whose association entry is a null reference
  for (const char* order : {"pt", "key"}) {
    edm::test::TestProcessor tester(
        producerConfig(std::string(",\n  daughterOrder = cms.string('") + order + "')", "", true));
    auto event = tester.test();
    auto const& out = *event.get<std::vector<pat::GenJet>>();
    REQUIRE(out.size() == 4);
    auto const& j30 = jetWithPt(out, 30.);
    CHECK(j30.originalObject()->numberOfDaughters() == 3);
    CHECK(j30.numberOfDaughters() == 2);  // the unmapped one is dropped, no null Ptr is stored
    for (size_t i = 0; i < j30.numberOfDaughters(); ++i)
      CHECK(j30.daughterPtr(i).isNonnull());
    CHECK(keys(j30) == std::vector<unsigned>{5, 6});  // gens {3, 2} -> packed {5, 6}
    // the other jets are untouched
    CHECK(jetWithPt(out, 80.).numberOfDaughters() == 3);
    CHECK(patgenjettest::badBackRefs(out) == 0);
  }
}

TEST_CASE("PATGenJetProducer: user data with stripped postfix", "[PATGenJetProducer]") {
  const std::string extra = ",\n  userData = " + patgenjettest::userData("cms.InputTag('" + kInputs + "', 'tau1')",
                                                                         "cms.InputTag('" + kInputs + "', 'nconst')",
                                                                         "'ForPatGenJets'");
  edm::test::TestProcessor tester(producerConfig(extra));
  auto event = tester.test();
  auto const& out = *event.get<std::vector<pat::GenJet>>();
  REQUIRE(out.size() == 4);
  for (auto const& j : out) {
    const unsigned inputIndex = j.originalObjectRef().key();
    // label = encoded InputTag with the postfix stripped (PATUserDataMerger.h:104-118)
    CHECK(j.hasUserFloat("genInputs:tau1"));
    CHECK_FALSE(j.hasUserFloat("genInputsForPatGenJets:tau1"));
    // value from the map
    CHECK(j.userFloat("genInputs:tau1") == Approx(0.1 * (inputIndex + 1)));
    CHECK(j.userInt("genInputs:nconst") == static_cast<int>(j.originalObject()->numberOfDaughters()));
  }
}

namespace {
  // genJetForwardMap: input jets -> "fwdjets" (pt 80 -> key 2, 50 -> 0, 30 -> 1, 12 -> key 3 or, with dropPartners, none)
  void checkGenJetForwardMap(bool drop) {
    edm::test::TestProcessor tester(
        producerConfig(",\n  genJetForwardMap = cms.InputTag('" + kInputs + ":jetmap')", "", false, drop));
    auto event = tester.test();
    auto const& out = *event.get<std::vector<pat::GenJet>>();
    REQUIRE(out.size() == 4);
    // the back ref is unchanged by the map (after the pt sort too)
    CHECK(patgenjettest::badBackRefs(out) == 0);
    const std::vector<double> fwdPt = {50., 30., 80., 12.};
    for (auto const& j : out) {
      auto const& l = j.genJetFwdRef();
      const int key = patgenjettest::expectedJetFwdKey(j.pt(), drop);
      INFO("jet pt " << j.pt());
      if (key >= 0) {
        REQUIRE(l.ref().isNonnull());
        CHECK(l.ref().key() == static_cast<unsigned>(key));
        CHECK(l.ref().id() != l.backRef().id());
        CHECK(l.ref()->pt() == Approx(fwdPt[key]));  // follows the map to the right jet
        CHECK(l.backRef()->pt() == Approx(j.pt()));
      } else {  // no partner: forward = back
        CHECK(l.backRef().isNonnull());
        CHECK(l.ref() == l.backRef());
      }
    }
  }
}  // namespace

TEST_CASE("PATGenJetProducer: genJetForwardMap", "[PATGenJetProducer]") { checkGenJetForwardMap(false); }

TEST_CASE("PATGenJetProducer: genJetForwardMap, jet without partner keeps forward = back", "[PATGenJetProducer]") {
  checkGenJetForwardMap(true);
}

namespace {
  const std::string kTwoMaps = ",\n  embedGenCandidates = cms.bool(True), genCandidatesForwardMaps = cms.VInputTag('" +
                               kInputs + ":map1', '" + kInputs + ":map2')";

  // the generator particles (input constituents) of the jet, in the order of the input jet
  std::vector<unsigned> constituentGens(int pt) {
    if (pt == 80)
      return {1, 5, 8, 9};
    if (pt == 50)
      return {7, 0};
    if (pt == 30)
      return {3, 2, 10};  // gen 10 only exists with the unmappedConstituent option
    return {4, 6};
  }
  int ptName(pat::GenJet const& j) { return static_cast<int>(std::lround(j.pt())); }
  double genPt(unsigned k) { return k < 10 ? patgenjettest::kGenPtExpected[k] : 7.; }
}  // namespace

TEST_CASE("PATGenJetProducer: embedGenCandidates False stores nothing", "[PATGenJetProducer]") {
  edm::test::TestProcessor tester(
      producerConfig(",\n  genCandidatesForwardMaps = cms.VInputTag('" + kInputs + ":map1')"));
  auto event = tester.test();
  auto const& out = *event.get<std::vector<pat::GenJet>>();
  REQUIRE(out.size() == 4);
  for (auto const& j : out) {
    CHECK(j.genCandidatesFwdPtr().empty());
    CHECK(j.genCandidates().empty());
  }
}

TEST_CASE("PATGenJetProducer: embedGenCandidates without maps", "[PATGenJetProducer]") {
  edm::test::TestProcessor tester(producerConfig(",\n  embedGenCandidates = cms.bool(True)", "", true));
  auto event = tester.test();
  auto const& out = *event.get<std::vector<pat::GenJet>>();
  REQUIRE(out.size() == 4);
  for (auto const& j : out) {
    INFO("jet pt " << j.pt());
    const auto gens = constituentGens(ptName(j));
    auto const& fp = j.genCandidatesFwdPtr();
    // one entry per generator particle of the input jet, in its order, also the one without packed partner
    REQUIRE(fp.size() == gens.size());
    REQUIRE(j.genCandidates().size() == gens.size());
    const edm::ProductID origId = j.originalObject()->sourceCandidatePtr(0).id();  // the genParticles product
    for (size_t i = 0; i < gens.size(); ++i) {
      // back = the original constituent, not re-keyed to packed
      CHECK(fp[i].backPtr().id() == origId);
      CHECK(fp[i].backPtr().key() == gens[i]);
      CHECK(fp[i].backPtr()->pt() == Approx(genPt(gens[i])));
      // no maps: forward = back
      CHECK(fp[i].ptr() == fp[i].backPtr());
      CHECK(j.genCandidates()[i] == fp[i].backPtr());
    }
    // the daughters are re-keyed to packed, independent of the stored generator particles
    for (size_t i = 0; i < j.numberOfDaughters(); ++i)
      CHECK(j.daughterPtr(i).id() != origId);
  }
}

namespace {
  // chain map1 -> map2: gens {1, 5, 8, 9} etc. forwarded to stage2; 11 gens (the unmapped option), drop: no partners
  void checkChain(bool drop) {
    edm::test::TestProcessor tester(producerConfig(kTwoMaps, "", true, drop));
    auto event = tester.test();
    auto const& out = *event.get<std::vector<pat::GenJet>>();
    REQUIRE(out.size() == 4);
    unsigned nNull = 0, nFwd = 0;
    for (auto const& j : out) {
      INFO("jet pt " << j.pt());
      const auto gens = constituentGens(ptName(j));
      auto const& fp = j.genCandidatesFwdPtr();
      REQUIRE(fp.size() == gens.size());
      for (size_t i = 0; i < gens.size(); ++i) {
        INFO("gen " << gens[i]);
        // back unchanged
        CHECK(fp[i].backPtr().key() == gens[i]);
        CHECK(fp[i].backPtr().id() == j.originalObject()->sourceCandidatePtr(0).id());
        const int key = patgenjettest::expectedStage2Key(gens[i], 11, drop);
        if (key < 0) {  // no partner in map1 or in map2: forward = back
          CHECK(fp[i].ptr() == fp[i].backPtr());
          ++nNull;
        } else {
          REQUIRE(fp[i].ptr().isNonnull());
          CHECK(fp[i].ptr().key() == static_cast<unsigned>(key));
          CHECK(fp[i].ptr().id() != fp[i].backPtr().id());
          CHECK(fp[i].ptr()->pt() ==
                Approx(genPt(gens[i])));  // the copy of the same particle: the whole chain was applied
          ++nFwd;
        }
      }
    }
    // jets 80, 50, 30, 12: 4 + 2 + 3 + 2 entries; with dropPartners 2 + 2 + 1 + 0 have a partner after both maps
    CHECK(nFwd == (drop ? 5u : 11u));
    CHECK(nNull == (drop ? 6u : 0u));
  }

  // only map1: gen 8 is kept by map1 and dropped by map2
  void checkOneMap(bool drop) {
    edm::test::TestProcessor tester(producerConfig(
        ",\n  embedGenCandidates = cms.bool(True), genCandidatesForwardMaps = cms.VInputTag('" + kInputs + ":map1')",
        "",
        false,
        drop));
    auto event = tester.test();
    auto const& out = *event.get<std::vector<pat::GenJet>>();
    auto const& fp = jetWithPt(out, 80.).genCandidatesFwdPtr();
    REQUIRE(fp.size() == 4);  // gens 1, 5, 8, 9
    CHECK(fp[0].ptr().key() == 0);
    CHECK(fp[1].ptr().key() == 1);
    CHECK(fp[2].ptr().key() == 2);  // gen 8 -> stage1 2
    if (drop)
      CHECK(fp[3].ptr() == fp[3].backPtr());  // gen 9: no partner in map1: forward = back
    else
      CHECK(fp[3].ptr().key() == static_cast<unsigned>(patgenjettest::expectedStage1Key(9, 10)));
  }
}  // namespace

TEST_CASE("PATGenJetProducer: embedGenCandidates with a chain of two forward maps", "[PATGenJetProducer]") {
  checkChain(false);
}

TEST_CASE("PATGenJetProducer: embedGenCandidates with one forward map", "[PATGenJetProducer]") { checkOneMap(false); }

TEST_CASE("PATGenJetProducer: embedGenCandidates, chain with particles without partner", "[PATGenJetProducer]") {
  checkChain(true);
}

TEST_CASE("PATGenJetProducer: embedGenCandidates, one map, particle without partner", "[PATGenJetProducer]") {
  checkOneMap(true);
}

TEST_CASE("PATGenJetProducer: configuration checks", "[PATGenJetProducer]") {
  CHECK_THROWS_AS(edm::test::TestProcessor(producerConfig(",\n  daughterOrder = cms.string('eta')")), cms::Exception);
}
