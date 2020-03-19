#include "CommonTools/BaseParticlePropagator/interface/makeMuon.h"
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

static constexpr const double kMuonMass = 0.10566;

TEST_CASE("makeMuon tests", "[makeMuon]") {
  SECTION("muon at rest") {
    auto m1 = rawparticle::makeMuon(true, math::XYZTLorentzVector{}, math::XYZTLorentzVector{});

    REQUIRE(m1.charge() == -1.);
    REQUIRE(m1.mass() == kMuonMass);
    REQUIRE(m1.px() == 0.);
    REQUIRE(m1.py() == 0.);
    REQUIRE(m1.pz() == 0.);
    REQUIRE(m1.x() == 0.);
    REQUIRE(m1.y() == 0.);
    REQUIRE(m1.z() == 0.);
    REQUIRE(m1.t() == 0.);

    //NOTE: energy is incorrectly calculated!
    //REQUIRE(m1.e() == kMuonMass);
    REQUIRE(m1.e() == 0.);
  }

  SECTION("anti-muon at rest") {
    auto m1 = rawparticle::makeMuon(false, math::XYZTLorentzVector{}, math::XYZTLorentzVector{});

    REQUIRE(m1.charge() == 1.);
    REQUIRE(m1.mass() == kMuonMass);
    REQUIRE(m1.px() == 0.);
    REQUIRE(m1.py() == 0.);
    REQUIRE(m1.pz() == 0.);
    REQUIRE(m1.x() == 0.);
    REQUIRE(m1.y() == 0.);
    REQUIRE(m1.z() == 0.);
    REQUIRE(m1.t() == 0.);

    //NOTE: energy is incorrectly calculated!
    //REQUIRE(m1.e() == kMuonMass);
    REQUIRE(m1.e() == 0.);
  }

  SECTION("muon at rest, transposed") {
    auto m1 = rawparticle::makeMuon(true, math::XYZTLorentzVector{}, math::XYZTLorentzVector{1., 2., 3., 4.});

    REQUIRE(m1.charge() == -1.);
    REQUIRE(m1.mass() == kMuonMass);
    REQUIRE(m1.px() == 0.);
    REQUIRE(m1.py() == 0.);
    REQUIRE(m1.pz() == 0.);
    REQUIRE(m1.x() == 1.);
    REQUIRE(m1.y() == 2.);
    REQUIRE(m1.z() == 3.);
    REQUIRE(m1.t() == 4.);

    //NOTE: energy is incorrectly calculated!
    //REQUIRE(m1.e() == kMuonMass);
    REQUIRE(m1.e() == 0.);
  }

  SECTION("muon in motion") {
    auto m1 = rawparticle::makeMuon(true, math::XYZTLorentzVector{1., 2., 3., 8.}, math::XYZTLorentzVector{});

    REQUIRE(m1.charge() == -1.);
    REQUIRE(m1.mass() == kMuonMass);
    REQUIRE(m1.px() == 1.);
    REQUIRE(m1.py() == 2.);
    REQUIRE(m1.pz() == 3.);
    REQUIRE(m1.x() == 0.);
    REQUIRE(m1.y() == 0.);
    REQUIRE(m1.z() == 0.);
    REQUIRE(m1.t() == 0.);

    //NOTE: energy is incorrectly calculated!
    //REQUIRE(m1.e() == sqrt(kMuonMass*kMuonMass+m1.px()*m1.px()+m1.py()*m1.py()+m1.pz()*m1.pz()));
    REQUIRE(m1.e() == 8.);
  }
}
