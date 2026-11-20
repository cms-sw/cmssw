// Unit tests for the pat::Jet -> pat::GenJet link (pat::Jet::patGenJetRef, class version 4).
// Hand-made objects only: no event data, no conditions.

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/GenJet.h"

#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "TClass.h"
#include "TDataMember.h"

#include <vector>

// DataFormats/Math/interface/Error.h defines a STATIC_CHECK macro that clashes with Catch2
#undef STATIC_CHECK
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

namespace {
  // access to the embedded gen jet (protected; set in production only by the PAT producers)
  struct Peek : pat::Jet {
    static void embed(pat::Jet& j, const reco::GenJet& g) { (j.*(&Peek::genJet_)).push_back(g); }
  };

  reco::GenJet makeGenJet(double pt) {
    return reco::GenJet(
        reco::Particle::LorentzVector(pt, 0., 0., pt), reco::Particle::Point(0, 0, 0), reco::GenJet::Specific());
  }
}  // namespace

TEST_CASE("pat::Jet link to pat::GenJet", "[patJet][patGenJet]") {
  // old collection (reco::GenJet) and its pat::GenJet twin collection
  std::vector<reco::GenJet> old{makeGenJet(50.), makeGenJet(30.)};
  edm::OrphanHandle<std::vector<reco::GenJet>> hOld(&old, edm::ProductID(1, 1));
  std::vector<pat::GenJet> twins;
  for (auto const& g : old)
    twins.emplace_back(g);
  edm::OrphanHandle<std::vector<pat::GenJet>> hTwins(&twins, edm::ProductID(1, 2));

  SECTION("default: no link, genJet() as before") {
    pat::Jet jet;
    CHECK(jet.patGenJetRef().isNull());
    CHECK(jet.patGenJet() == nullptr);
    CHECK(jet.genJet() == nullptr);
  }

  SECTION("only the old link: genJet() follows genJetFwdRef") {
    pat::Jet jet;
    edm::Ref<std::vector<reco::GenJet>> r(hOld, 1);
    jet.setGenJetRef(edm::FwdRef<std::vector<reco::GenJet>>(r, r));
    CHECK(jet.genJet() == &old[1]);
    CHECK(jet.patGenJet() == nullptr);
  }

  SECTION("pat::GenJet link set: genJet() follows it, before genJetFwdRef") {
    pat::Jet jet;
    edm::Ref<std::vector<reco::GenJet>> r(hOld, 1);
    jet.setGenJetRef(edm::FwdRef<std::vector<reco::GenJet>>(r, r));
    jet.setPatGenJetRef(pat::GenJetRef(hTwins, 1));
    CHECK(jet.patGenJetRef().key() == 1);
    CHECK(jet.patGenJet() == &twins[1]);
    CHECK(jet.genJet() == static_cast<const reco::GenJet*>(&twins[1]));
    CHECK(jet.genJet()->pt() == Catch::Approx(30.));
    // the keys of both links are the same (the twin sits at the same index as the old gen jet)
    CHECK(jet.patGenJetRef().key() == jet.genJetFwdRef().backRef().key());
  }

  SECTION("matchedGenJetPtr: pat::GenJet link, else the back link of genJetFwdRef, else null; same key") {
    pat::Jet none;
    CHECK(none.matchedGenJetPtr().isNull());
    pat::Jet oldOnly;
    edm::Ref<std::vector<reco::GenJet>> r(hOld, 1);
    oldOnly.setGenJetRef(edm::FwdRef<std::vector<reco::GenJet>>(r, r));
    CHECK(oldOnly.matchedGenJetPtr().id() == hOld.id());
    CHECK(oldOnly.matchedGenJetPtr().key() == 1);
    CHECK(oldOnly.matchedGenJetPtr().get() == &old[1]);
    pat::Jet both(oldOnly);
    both.setPatGenJetRef(pat::GenJetRef(hTwins, 1));
    CHECK(both.matchedGenJetPtr().id() == hTwins.id());
    CHECK(both.matchedGenJetPtr().key() == oldOnly.matchedGenJetPtr().key());
    CHECK(both.matchedGenJetPtr().get() == static_cast<const reco::GenJet*>(&twins[1]));
  }

  SECTION("an embedded gen jet still comes first") {
    pat::Jet jet;
    Peek::embed(jet, makeGenJet(70.));
    jet.setPatGenJetRef(pat::GenJetRef(hTwins, 0));
    REQUIRE(jet.genJet() != nullptr);
    CHECK(jet.genJet()->pt() == Catch::Approx(70.));
  }

  SECTION("the link survives a copy (updaters and selectors copy pat::Jets)") {
    pat::Jet jet;
    jet.setPatGenJetRef(pat::GenJetRef(hTwins, 0));
    pat::Jet copy(jet);
    CHECK(copy.patGenJet() == &twins[0]);
  }
}

TEST_CASE("pat::Jet class version 4 stores the link", "[patJet][dictionary]") {
  TClass* cl = TClass::GetClass("pat::io_v1::Jet");
  REQUIRE(cl != nullptr);
  CHECK(cl->GetClassVersion() == 4);
  auto* member = cl->GetDataMember("patGenJetRef_");
  REQUIRE(member != nullptr);
  CHECK(member->IsPersistent());
}
