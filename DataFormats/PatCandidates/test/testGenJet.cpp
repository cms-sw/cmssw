// Unit tests for pat::GenJet.
// Hand-made objects only (input "H"): no event data, no conditions.

#include "DataFormats/PatCandidates/interface/GenJet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TClass.h"
#include "TFile.h"
#include "TList.h"
#include "TStreamerInfo.h"
#include "TTree.h"

#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// DataFormats/Math/interface/Error.h defines a STATIC_CHECK macro that clashes with Catch2
#undef STATIC_CHECK
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

using Catch::Approx;

namespace {
  // Access to the transient caches (protected) for the "caches reset after read" check.
  struct Peek : pat::GenJet {
    static bool daughtersCached(const pat::GenJet& j) { return (j.*(&Peek::daughtersTemp_)).isSet(); }
    static bool genCandsCached(const pat::GenJet& j) { return (j.*(&Peek::genCandidatesTemp_)).isSet(); }
  };

  const edm::ProductID kConstituentsPID(1, 1);
  const edm::ProductID kGenJetsPID(1, 2);
  const edm::ProductID kSubjetsPID(1, 3);
  const edm::ProductID kGenPartPID(1, 4);  // generator particles (genCandidates)
  const edm::ProductID kGenPartFwdPID(1, 5);

  reco::GenJet::Specific makeSpecific() {
    reco::GenJet::Specific s;
    s.m_EmEnergy = 30.f;
    s.m_HadEnergy = 60.f;
    s.m_InvisibleEnergy = 5.f;
    s.m_AuxiliaryEnergy = 1.f;
    s.m_ChargedHadronEnergy = 40.f;
    s.m_NeutralHadronEnergy = 20.f;
    s.m_ChargedEmEnergy = 4.f;
    s.m_NeutralEmEnergy = 26.f;
    s.m_MuonEnergy = 3.f;
    s.m_ChargedHadronMultiplicity = 7;
    s.m_NeutralHadronMultiplicity = 2;
    s.m_ChargedEmMultiplicity = 1;
    s.m_NeutralEmMultiplicity = 5;
    s.m_MuonMultiplicity = 1;
    return s;
  }

  // constituents live here so that the CandidatePtrs of the test jets stay valid
  std::vector<reco::GenParticle> const& constituents() {
    static const std::vector<reco::GenParticle> parts = [] {
      std::vector<reco::GenParticle> v;
      const double energies[] = {50., 30., 15., 5.};
      for (double e : energies) {
        reco::Particle::LorentzVector p4(e, 0., 0., e);  // transverse: Et = E, as nCarrying uses Et
        v.emplace_back(0, p4, reco::Particle::Point(0, 0, 0), 22, 1, true);
      }
      return v;
    }();
    return parts;
  }

  reco::GenJet makeRecoGenJet(double pt = 50., double eta = 1.0, double phi = 0.5, double mass = 10.) {
    reco::Particle::PolarLorentzVector pp4(pt, eta, phi, mass);
    reco::Particle::LorentzVector p4(pp4.px(), pp4.py(), pp4.pz(), pp4.energy());
    reco::GenJet rj(p4, reco::Particle::Point(0.1, 0.2, 0.3), makeSpecific());
    rj.setJetArea(0.5);
    auto const& parts = constituents();
    for (size_t i = 0; i < parts.size(); ++i)
      rj.addDaughter(reco::CandidatePtr(kConstituentsPID, &parts[i], i));
    return rj;
  }

}  // namespace

// --------------------------------------------------------------------------------------------------
TEST_CASE("construction from reco::GenJet keeps kinematics and Specific", "[GenJet][construction]") {
  const reco::GenJet rj = makeRecoGenJet();
  static_assert(std::is_convertible_v<const reco::GenJet&, pat::GenJet>, "implicit constructor");
  pat::GenJet j = rj;  // implicit, as pat::Jet

  CHECK(j.p4() == rj.p4());
  CHECK(j.vertex() == rj.vertex());
  CHECK(j.jetArea() == Approx(0.5));
  CHECK(j.numberOfDaughters() == constituents().size());
  CHECK(j.emEnergy() == 30.f);
  CHECK(j.hadEnergy() == 60.f);
  CHECK(j.invisibleEnergy() == 5.f);
  CHECK(j.auxiliaryEnergy() == 1.f);
  CHECK(j.getSpecific().m_NeutralEmMultiplicity == 5);

  SECTION("Ptr and RefToBase constructors") {
    std::vector<reco::GenJet> coll{rj};
    edm::Ptr<reco::GenJet> ptr(kGenJetsPID, &coll[0], 0);
    pat::GenJet fromPtr = ptr;
    CHECK(fromPtr.p4() == rj.p4());
    CHECK(fromPtr.originalObjectRef().id() == kGenJetsPID);

    std::vector<pat::GenJet> pcoll{j};
    edm::Ptr<pat::GenJet> pptr(kGenJetsPID, &pcoll[0], 0);
    pat::GenJet fromPatPtr = pptr;
    CHECK(fromPatPtr.p4() == rj.p4());
    CHECK(fromPatPtr.originalObjectRef().key() == 0);
  }
}

// ---------------------------------------------------------------- behaviour groups, one block per group
TEST_CASE("2.3 JEC: factor 1, copy, empty lists, ERROR, NONE", "[GenJet][2.3][JEC]") {
  pat::GenJet j(makeRecoGenJet());
  // row: jecFactor(any) = 1
  CHECK(j.jecFactor("Uncorrected") == 1.f);
  CHECK(j.jecFactor("L3Absolute", "none", "") == 1.f);
  CHECK(j.jecFactor("L5Flavor", "bottom", "otherSet") == 1.f);
  CHECK(j.jecFactor(0) == 1.f);
  CHECK(j.jecFactor(3, pat::JetCorrFactors::BOTTOM, 2) == 1.f);
  // row: correctedJet / correctedP4
  CHECK(j.correctedP4("Uncorrected") == j.p4());
  CHECK(j.correctedP4(0) == j.p4());
  CHECK(j.correctedJet("Uncorrected").p4() == j.p4());
  // row: availableJECLevels(int) empty
  CHECK(j.availableJECLevels(0).empty());
  CHECK(j.availableJECLevels(5).empty());
  // row: availableJECSets, availableJECLevels(string), jecSet(s)Available
  CHECK(j.availableJECSets().empty());
  CHECK(j.availableJECLevels(std::string("patJetCorrFactors")).empty());
  CHECK_FALSE(j.jecSetsAvailable());
  CHECK_FALSE(j.jecSetAvailable(std::string("patJetCorrFactors")));
  CHECK_FALSE(j.jecSetAvailable(0u));
  // row: currentJECSet / currentJECLevel
  CHECK(j.currentJECSet() == "ERROR");
  CHECK(j.currentJECLevel() == "ERROR");
  // row: currentJECFlavor defined NONE
  CHECK(j.currentJECFlavor() == pat::JetCorrFactors::NONE);
  // rawFactor column expression: 1 - jecFactor('Uncorrected') = 0
  CHECK(1.f - j.jecFactor("Uncorrected") == 0.f);
}

TEST_CASE("2.3 PF-named: inherited 8 equal the Specific", "[GenJet][2.3][PF]") {
  const auto s = makeSpecific();
  pat::GenJet j(makeRecoGenJet());
  CHECK(j.chargedHadronEnergy() == s.m_ChargedHadronEnergy);
  CHECK(j.neutralHadronEnergy() == s.m_NeutralHadronEnergy);
  CHECK(j.chargedEmEnergy() == s.m_ChargedEmEnergy);
  CHECK(j.neutralEmEnergy() == s.m_NeutralEmEnergy);
  CHECK(j.muonEnergy() == s.m_MuonEnergy);
  CHECK(j.chargedHadronMultiplicity() == s.m_ChargedHadronMultiplicity);
  CHECK(j.neutralHadronMultiplicity() == s.m_NeutralHadronMultiplicity);
  CHECK(j.muonMultiplicity() == s.m_MuonMultiplicity);
}

TEST_CASE("2.3 PF-named: fractions and mapped members equal their formulas", "[GenJet][2.3][PF]") {
  const auto s = makeSpecific();
  pat::GenJet j(makeRecoGenJet());
  const double E = j.energy();
  REQUIRE(E > 0);
  // fractions (copy of pat::Jet with jecSetsAvailable() false)
  CHECK(j.chargedHadronEnergyFraction() == Approx(s.m_ChargedHadronEnergy / E));
  CHECK(j.neutralHadronEnergyFraction() == Approx(s.m_NeutralHadronEnergy / E));
  CHECK(j.chargedEmEnergyFraction() == Approx(s.m_ChargedEmEnergy / E));
  CHECK(j.neutralEmEnergyFraction() == Approx(s.m_NeutralEmEnergy / E));
  CHECK(j.muonEnergyFraction() == Approx(s.m_MuonEnergy / E));
  // mapped
  CHECK(j.photonEnergy() == s.m_NeutralEmEnergy);
  CHECK(j.photonEnergyFraction() == Approx(s.m_NeutralEmEnergy / E));
  CHECK(j.electronEnergy() == s.m_ChargedEmEnergy);
  CHECK(j.electronEnergyFraction() == Approx(s.m_ChargedEmEnergy / E));
  CHECK(j.chargedMuEnergy() == s.m_MuonEnergy);
  CHECK(j.chargedMuEnergyFraction() == Approx(s.m_MuonEnergy / E));
  CHECK(j.photonMultiplicity() == s.m_NeutralEmMultiplicity);
  CHECK(j.electronMultiplicity() == s.m_ChargedEmMultiplicity);
  CHECK(j.chargedMultiplicity() == s.m_ChargedHadronMultiplicity + s.m_ChargedEmMultiplicity + s.m_MuonMultiplicity);
  CHECK(j.neutralMultiplicity() == s.m_NeutralHadronMultiplicity + s.m_NeutralEmMultiplicity);
}

TEST_CASE("2.3 subjets: empty collection, groomedMass -1, counters", "[GenJet][2.3][subjets]") {
  pat::GenJet j(makeRecoGenJet());
  // rows: subjets(index), subjets(label) -> static empty collection
  CHECK_NOTHROW(j.subjets(0));
  CHECK(j.subjets(0).empty());
  CHECK(j.subjets(7).empty());
  CHECK(j.subjets("SoftDropPuppi").empty());
  // row: groomedMass -1
  CHECK(j.groomedMass() == -1.0);
  CHECK(j.groomedMass(3) == -1.0);
  CHECK(j.groomedMass("SoftDropPuppi") == -1.0);
  // row: hasSubjets / nSubjetCollections / subjetCollectionNames
  CHECK_FALSE(j.hasSubjets("SoftDropPuppi"));
  CHECK(j.nSubjetCollections() == 0);
  CHECK(j.subjetCollectionNames().empty());

  SECTION("an empty collection under a label also gives -1") {
    j.addSubjets(pat::GenJetPtrCollection(), "SoftDropPuppi");
    CHECK(j.hasSubjets("SoftDropPuppi"));
    CHECK(j.groomedMass("SoftDropPuppi") == -1.0);
    CHECK(j.groomedMass(0) == -1.0);
  }
  SECTION("with subjets: mass of the sum of subjet p4, label lookup") {
    std::vector<pat::GenJet> subs{pat::GenJet(makeRecoGenJet(30., 1.0, 0.4, 3.)),
                                  pat::GenJet(makeRecoGenJet(20., 1.1, 0.7, 2.))};
    pat::GenJetPtrCollection ptrs{edm::Ptr<pat::GenJet>(kSubjetsPID, &subs[0], 0),
                                  edm::Ptr<pat::GenJet>(kSubjetsPID, &subs[1], 1)};
    j.addSubjets(ptrs, "SoftDropPuppi");
    const double expected = (subs[0].p4() + subs[1].p4()).mass();
    CHECK(j.groomedMass("SoftDropPuppi") == Approx(expected));
    CHECK(j.groomedMass(0) == Approx(expected));
    CHECK(j.groomedMass("Pruned") == -1.0);
    CHECK(j.subjets("SoftDropPuppi").size() == 2);
    CHECK(j.subjets("SoftDropPuppi")[1].key() == 1);
    CHECK(j.nSubjetCollections() == 1);
    CHECK(j.subjetCollectionNames() == std::vector<std::string>{"SoftDropPuppi"});
    // daughters flattened through the subjets (copy of Jet.cc:616): the fat jet's own daughters are
    // GenParticles, not jets, so they stay as they are
    CHECK(j.numberOfDaughters() == constituents().size());
  }
  SECTION("with subjets and a null daughter (keepUnmappedDaughters): counted, not dereferenced") {
    pat::GenJet jn(makeRecoGenJet());
    const size_t n = jn.numberOfDaughters();
    jn.addDaughter(reco::CandidatePtr());  // the null slot PATGenJetSlimmer writes for an unmapped constituent
    std::vector<pat::GenJet> subs{pat::GenJet(makeRecoGenJet(30., 1.0, 0.4, 3.))};
    jn.addSubjets({edm::Ptr<pat::GenJet>(kSubjetsPID, &subs[0], 0)}, "SoftDropPuppi");
    CHECK_NOTHROW(jn.numberOfDaughters());
    CHECK(jn.numberOfDaughters() == n + 1);
    CHECK(jn.daughterPtr(n).isNull());
  }
}

TEST_CASE("2.3 gen matching: genJet() is the jet", "[GenJet][2.3][genJet]") {
  std::vector<pat::GenJet> coll{pat::GenJet(makeRecoGenJet(50.)), pat::GenJet(makeRecoGenJet(40.))};
  // row: genJet() returns this
  CHECK(coll[1].genJet() == &coll[1]);
  CHECK(static_cast<const reco::GenJet*>(&coll[0]) == coll[0].genJet());
}

TEST_CASE("genJetFwdRef: null, or the reco::GenJet the jet was built from", "[GenJet][genJetFwdRef]") {
  const reco::GenJet rj = makeRecoGenJet(42.);
  std::vector<reco::GenJet> rcoll{makeRecoGenJet(10.), rj};

  SECTION("null for default-constructed and for a jet built from a reco::GenJet object") {
    pat::GenJet def;
    CHECK(def.genJetFwdRef().isNull());
    CHECK(def.genJetFwdRef().backRef().isNull());
    pat::GenJet fromObj(rj);
    CHECK(fromObj.genJetFwdRef().isNull());
    CHECK(fromObj.genJetFwdRef().backRef().isNull());
  }

  SECTION("from edm::Ptr<reco::GenJet>: forward and back ref are the input") {
    edm::Ptr<reco::GenJet> ptr(kGenJetsPID, &rcoll[1], 1);
    pat::GenJet j(ptr);
    REQUIRE(j.genJetFwdRef().isNonnull());
    CHECK(j.genJetFwdRef().ref().id() == kGenJetsPID);
    CHECK(j.genJetFwdRef().ref().key() == 1);
    CHECK(j.genJetFwdRef().backRef().id() == kGenJetsPID);
    CHECK(j.genJetFwdRef().backRef().key() == 1);
    CHECK(j.genJetFwdRef()->pt() == Approx(42.));
  }

  SECTION("from edm::RefToBase<reco::GenJet>: forward and back ref are the input") {
    edm::OrphanHandle<std::vector<reco::GenJet>> rh(&rcoll, kGenJetsPID);
    edm::RefToBase<reco::GenJet> rref(edm::Ref<std::vector<reco::GenJet>>(rh, 1));
    pat::GenJet j(rref);
    REQUIRE(j.genJetFwdRef().isNonnull());
    CHECK(j.genJetFwdRef().ref().id() == kGenJetsPID);
    CHECK(j.genJetFwdRef().ref().key() == 1);
    CHECK(j.genJetFwdRef().backRef().id() == kGenJetsPID);
    CHECK(j.genJetFwdRef().backRef().key() == 1);
  }

  SECTION("the pat::GenJet Ptr and RefToBase constructors keep it") {
    pat::GenJet orig{edm::Ptr<reco::GenJet>(kGenJetsPID, &rcoll[1], 1)};
    std::vector<pat::GenJet> pcoll{pat::GenJet(makeRecoGenJet(11.)), orig};
    edm::Ptr<pat::GenJet> pptr(kSubjetsPID, &pcoll[1], 1);
    pat::GenJet fromPtr(pptr);
    CHECK(fromPtr.genJetFwdRef().ref().id() == kGenJetsPID);
    CHECK(fromPtr.genJetFwdRef().ref().key() == 1);
    CHECK(fromPtr.genJetFwdRef().backRef().id() == kGenJetsPID);
    CHECK(fromPtr.genJetFwdRef().backRef().key() == 1);

    edm::OrphanHandle<std::vector<pat::GenJet>> ph(&pcoll, kSubjetsPID);
    edm::RefToBase<pat::GenJet> pref(edm::Ref<std::vector<pat::GenJet>>(ph, 1));
    pat::GenJet fromRef(pref);
    CHECK(fromRef.genJetFwdRef().ref().id() == kGenJetsPID);
    CHECK(fromRef.genJetFwdRef().ref().key() == 1);
    CHECK(fromRef.genJetFwdRef().backRef().id() == kGenJetsPID);
    CHECK(fromRef.genJetFwdRef().backRef().key() == 1);
    // plain copy as well
    pat::GenJet copy(orig);
    CHECK(copy.genJetFwdRef().backRef().key() == 1);
  }

  SECTION("updateFwdGenJetFwdRef changes only the forward ref") {
    pat::GenJet j{edm::Ptr<reco::GenJet>(kGenJetsPID, &rcoll[1], 1)};
    const edm::ProductID newPID(1, 20);
    j.updateFwdGenJetFwdRef(reco::GenJetRef(newPID, &rcoll[0], 0));
    CHECK(j.genJetFwdRef().ref().id() == newPID);
    CHECK(j.genJetFwdRef().ref().key() == 0);
    CHECK(j.genJetFwdRef().backRef().id() == kGenJetsPID);
    CHECK(j.genJetFwdRef().backRef().key() == 1);
  }
}

TEST_CASE("genCandidates: empty by default, append, forward-else-back, update", "[GenJet][genCandidates]") {
  // generator particles that the (transient) Ptrs point to
  std::vector<reco::GenParticle> backs(3), fwds(3);
  auto back = [&](size_t i) { return reco::GenParticlePtr(kGenPartPID, &backs[i], i); };
  auto fwd = [&](size_t i) { return reco::GenParticlePtr(kGenPartFwdPID, &fwds[i], i); };
  // non-null Ptr to a product that is not there (no pointer to it, no product getter): isAvailable() is false
  auto unavailable = [&](size_t i) {
    return reco::GenParticlePtr(kGenPartFwdPID, static_cast<const reco::GenParticle*>(nullptr), i);
  };

  pat::GenJet j(makeRecoGenJet());
  CHECK(j.genCandidates().empty());
  CHECK(j.genCandidatesFwdPtr().empty());
  // out of range on an empty jet: no-op, no throw
  CHECK_NOTHROW(j.updateFwdGenCandidateFwdPtr(0, fwd(0)));
  CHECK(j.genCandidatesFwdPtr().empty());

  SECTION("setGenCandidates appends and resets the cache") {
    reco::GenParticleFwdPtrVector first{reco::GenParticleFwdPtr(fwd(0), back(0)),
                                        reco::GenParticleFwdPtr(fwd(1), back(1))};
    j.setGenCandidates(first);
    REQUIRE(j.genCandidatesFwdPtr().size() == 2);
    REQUIRE(j.genCandidates().size() == 2);
    REQUIRE(Peek::genCandsCached(j));
    j.setGenCandidates(reco::GenParticleFwdPtrVector{reco::GenParticleFwdPtr(fwd(2), back(2))});
    CHECK_FALSE(Peek::genCandsCached(j));
    REQUIRE(j.genCandidatesFwdPtr().size() == 3);
    CHECK(j.genCandidatesFwdPtr()[2].backPtr().key() == 2);
    CHECK(j.genCandidates().size() == 3);
    // the daughters (generator constituents) are unaffected
    CHECK(j.numberOfDaughters() == constituents().size());
  }

  SECTION("genCandidates() holds fwdptr.ptr() (as pat::Jet); the FwdPtr itself falls back to the back Ptr") {
    reco::GenParticleFwdPtrVector v;
    v.emplace_back(fwd(0), back(0));          // forward available
    v.emplace_back(unavailable(1), back(1));  // forward not available
    j.setGenCandidates(v);
    auto const& c = j.genCandidates();
    REQUIRE(c.size() == 2);
    CHECK(c[0].id() == kGenPartFwdPID);
    CHECK(c[0].get() == &fwds[0]);
    // the cache copies FwdPtr::ptr(), the forward Ptr, whether or not it is available ...
    CHECK(c[1].id() == kGenPartFwdPID);
    CHECK_FALSE(c[1].isAvailable());
    // ... while dereferencing the stored FwdPtr itself uses the back Ptr if the forward one is unavailable
    CHECK(j.genCandidatesFwdPtr()[0].get() == &fwds[0]);
    CHECK(j.genCandidatesFwdPtr()[1].get() == &backs[1]);
    CHECK(j.genCandidatesFwdPtr()[1].backPtr().id() == kGenPartPID);
  }

  SECTION("updateFwdGenCandidateFwdPtr in range: forward changed, back kept, cache refreshed") {
    reco::GenParticleFwdPtrVector v;
    v.emplace_back(fwd(0), back(0));
    v.emplace_back(unavailable(1), back(1));
    j.setGenCandidates(v);
    REQUIRE_FALSE(j.genCandidates()[1].isAvailable());
    REQUIRE(Peek::genCandsCached(j));

    const edm::ProductID newPID(1, 21);
    j.updateFwdGenCandidateFwdPtr(1, reco::GenParticlePtr(newPID, &fwds[2], 2));
    CHECK_FALSE(Peek::genCandsCached(j));
    CHECK(j.genCandidatesFwdPtr()[1].ptr().id() == newPID);
    CHECK(j.genCandidatesFwdPtr()[1].ptr().key() == 2);
    CHECK(j.genCandidatesFwdPtr()[1].backPtr().id() == kGenPartPID);
    CHECK(j.genCandidatesFwdPtr()[1].backPtr().key() == 1);
    CHECK(j.genCandidatesFwdPtr()[0].ptr().id() == kGenPartFwdPID);  // other entry untouched
    CHECK(j.genCandidates()[1].id() == newPID);                      // rebuilt cache sees the new pointer
    CHECK(j.genCandidates()[1].get() == &fwds[2]);
    CHECK(j.genCandidates()[1].isAvailable());
    CHECK(j.genCandidates()[0].id() == kGenPartFwdPID);
  }

  SECTION("updateFwdGenCandidateFwdPtr out of range: no-op") {
    j.setGenCandidates(reco::GenParticleFwdPtrVector{reco::GenParticleFwdPtr(fwd(0), back(0))});
    REQUIRE(j.genCandidates().size() == 1);
    REQUIRE(Peek::genCandsCached(j));
    CHECK_NOTHROW(j.updateFwdGenCandidateFwdPtr(1, fwd(2)));
    CHECK(Peek::genCandsCached(j));  // untouched, cache not reset
    REQUIRE(j.genCandidatesFwdPtr().size() == 1);
    CHECK(j.genCandidatesFwdPtr()[0].ptr().id() == kGenPartFwdPID);
    CHECK(j.genCandidatesFwdPtr()[0].ptr().key() == 0);
    CHECK(j.genCandidates().size() == 1);
  }
}

TEST_CASE("2.3 jetCharge, genParton", "[GenJet][2.3][jetCharge]") {
  pat::GenJet j(makeRecoGenJet());
  CHECK(j.jetCharge() == 0.f);
  CHECK(j.genParton() == nullptr);
  j.setJetCharge(1.f);
  CHECK(j.jetCharge() == 1.f);
}

TEST_CASE("2.3 generic PAT lookups keep throwing", "[GenJet][2.3][PATObject]") {
  pat::GenJet j(makeRecoGenJet());
  CHECK_THROWS_AS(j.userFloat("pileupJetIdPuppi:fullDiscriminant"), cms::Exception);
  CHECK_THROWS_AS(j.userInt("hfcentralEtaStripSize"), cms::Exception);
  CHECK_THROWS(j.efficiency("trigger"));
  CHECK_THROWS(j.resolEt());
  CHECK_FALSE(j.hasUserFloat("x"));
  j.addUserFloat("x", 1.5f);
  CHECK(j.userFloat("x") == 1.5f);
}

TEST_CASE("scaleEnergy rescales p4 only", "[GenJet][kinematics]") {
  pat::GenJet j(makeRecoGenJet());
  const auto p4 = j.p4();
  j.scaleEnergy(2.0);
  CHECK(j.energy() == Approx(2.0 * p4.energy()));
  CHECK(j.jecFactor("Uncorrected") == 1.f);
  CHECK(j.chargedHadronEnergy() == 40.f);
}

TEST_CASE("copy and correctedJet equal the jet", "[GenJet][copy]") {
  pat::GenJet j(makeRecoGenJet());
  j.setJetCharge(2.f);
  j.addUserFloat("NjettinessAK8Puppi:tau1", 0.3f);
  j.addUserInt("nConst", 4);
  std::vector<pat::GenJet> subs{pat::GenJet(makeRecoGenJet(30.))};
  j.addSubjets({edm::Ptr<pat::GenJet>(kSubjetsPID, &subs[0], 0)}, "SoftDropPuppi");

  auto same = [&](const pat::GenJet& c) {
    CHECK(c.p4() == j.p4());
    CHECK(c.vertex() == j.vertex());
    CHECK(c.jetArea() == j.jetArea());
    CHECK(c.chargedHadronEnergy() == j.chargedHadronEnergy());
    CHECK(c.neutralEmMultiplicity() == j.neutralEmMultiplicity());
    CHECK(c.jetCharge() == 2.f);
    CHECK(c.userFloat("NjettinessAK8Puppi:tau1") == 0.3f);
    CHECK(c.userInt("nConst") == 4);
    CHECK(c.subjets("SoftDropPuppi").size() == 1);
    CHECK(c.groomedMass("SoftDropPuppi") == Approx(j.groomedMass("SoftDropPuppi")));
    CHECK(c.numberOfDaughters() == j.numberOfDaughters());
  };
  pat::GenJet copy(j);
  same(copy);
  pat::GenJet assigned;
  assigned = j;
  same(assigned);
  same(j.correctedJet("Uncorrected"));
  same(j.correctedJet(0));
  std::unique_ptr<pat::GenJet> cl(j.clone());
  same(*cl);
}

// ---------------------------------------------------------------- ROOT I/O
namespace {
  const char* const kRoundTripFile = "testGenJet_roundtrip.root";
  const char* const kVersionFile = "testGenJet_classversion.root";

  // the collection is returned by value: the transient cached pointers of the genJetFwdRef and
  // genCandidates Ptrs are not used after the move, only (product id, key) are checked
  std::vector<pat::GenJet> makeCollectionForIO(std::vector<pat::GenJet>& subjetStore) {
    subjetStore = {pat::GenJet(makeRecoGenJet(30.)), pat::GenJet(makeRecoGenJet(20.))};
    static const std::vector<reco::GenParticle> genParts(4);
    static const std::vector<reco::GenJet> srcJets{makeRecoGenJet(120.)};
    // jet 0 is made from a reco::GenJet (genJetFwdRef set), jet 1 from an object (genJetFwdRef null)
    std::vector<pat::GenJet> coll{pat::GenJet(edm::Ptr<reco::GenJet>(kGenJetsPID, &srcJets[0], 3)),
                                  pat::GenJet(makeRecoGenJet(60.))};
    reco::GenParticleFwdPtrVector gc;
    gc.emplace_back(reco::GenParticlePtr(kGenPartFwdPID, &genParts[2], 2),
                    reco::GenParticlePtr(kGenPartPID, &genParts[3], 3));
    coll[0].setGenCandidates(gc);
    coll[0].addSubjets({edm::Ptr<pat::GenJet>(kSubjetsPID, &subjetStore[0], 0),
                        edm::Ptr<pat::GenJet>(kSubjetsPID, &subjetStore[1], 1)},
                       "SoftDropPuppi");
    coll[0].addUserFloat("NjettinessAK8Puppi:tau1", 0.31f);
    coll[0].addUserInt("hfcentralEtaStripSize", 2);
    coll[0].setJetCharge(1.f);
    // fill the transient caches before writing
    (void)coll[0].numberOfDaughters();
    (void)coll[0].genCandidates();
    return coll;
  }

  void writeCollection(const char* fname, const std::vector<pat::GenJet>& coll) {
    std::unique_ptr<TFile> f(TFile::Open(fname, "RECREATE"));
    REQUIRE(f);
    TTree t("t", "pat::GenJet test");
    const std::vector<pat::GenJet>* ptr = &coll;
    REQUIRE(t.Branch("jets", &ptr) != nullptr);
    REQUIRE(t.Fill() > 0);
    t.Write();
    f->Close();
  }

  std::vector<pat::GenJet> readCollection(const char* fname) {
    std::unique_ptr<TFile> f(TFile::Open(fname, "READ"));
    REQUIRE(f);
    TTree* t = f->Get<TTree>("t");
    REQUIRE(t);
    std::vector<pat::GenJet>* ptr = nullptr;
    t->SetBranchAddress("jets", &ptr);
    REQUIRE(t->GetEntry(0) > 0);
    REQUIRE(ptr);
    std::vector<pat::GenJet> out(*ptr);
    delete ptr;
    return out;
  }
}  // namespace

TEST_CASE("ROOT write/read of std::vector<pat::GenJet>", "[GenJet][io]") {
  std::vector<pat::GenJet> subjetStore;
  const auto coll = makeCollectionForIO(subjetStore);
  REQUIRE(Peek::daughtersCached(coll[0]));
  REQUIRE(Peek::genCandsCached(coll[0]));
  writeCollection(kRoundTripFile, coll);
  const auto back = readCollection(kRoundTripFile);

  REQUIRE(back.size() == 2);
  // caches are transient and reset by the ioread rules
  CHECK_FALSE(Peek::daughtersCached(back[0]));
  CHECK_FALSE(Peek::genCandsCached(back[0]));
  for (size_t i = 0; i < 2; ++i) {
    CHECK(back[i].p4() == coll[i].p4());
    CHECK(back[i].vertex().x() == Approx(coll[i].vertex().x()));  // vertex is stored with reduced precision
    CHECK(back[i].vertex().z() == Approx(coll[i].vertex().z()));
    CHECK(back[i].jetArea() == coll[i].jetArea());
    // Specific survives (the reason for PATObject<reco::GenJet>)
    CHECK(back[i].chargedHadronEnergy() == 40.f);
    CHECK(back[i].neutralEmMultiplicity() == 5);
    CHECK(back[i].emEnergy() == 30.f);
    // stored daughters: product id and key. The stored vector is read without the subjet flattening of
    // daughterPtrVector(), which dereferences the daughters and therefore needs the event (as in pat::Jet).
    auto const& daus = back[i].reco::CompositePtrCandidate::daughterPtrVector();
    REQUIRE(daus.size() == constituents().size());
    CHECK(daus[2].id() == kConstituentsPID);
    CHECK(daus[2].key() == 2);
    CHECK(back[i].genJet() == &back[i]);
  }
  // genJetFwdRef: id and key of forward and back ref; null for the jet made from an object
  CHECK(back[0].genJetFwdRef().ref().id() == kGenJetsPID);
  CHECK(back[0].genJetFwdRef().ref().key() == 3);
  CHECK(back[0].genJetFwdRef().backRef().id() == kGenJetsPID);
  CHECK(back[0].genJetFwdRef().backRef().key() == 3);
  CHECK(back[1].genJetFwdRef().isNull());
  // genCandidates: forward and back Ptr (id, key)
  REQUIRE(back[0].genCandidatesFwdPtr().size() == 1);
  CHECK(back[0].genCandidatesFwdPtr()[0].ptr().id() == kGenPartFwdPID);
  CHECK(back[0].genCandidatesFwdPtr()[0].ptr().key() == 2);
  CHECK(back[0].genCandidatesFwdPtr()[0].backPtr().id() == kGenPartPID);
  CHECK(back[0].genCandidatesFwdPtr()[0].backPtr().key() == 3);
  CHECK(back[1].genCandidatesFwdPtr().empty());
  CHECK(back[0].jetCharge() == 1.f);
  // subjets with label (id, key)
  CHECK(back[0].subjetCollectionNames() == std::vector<std::string>{"SoftDropPuppi"});
  REQUIRE(back[0].subjets("SoftDropPuppi").size() == 2);
  CHECK(back[0].subjets("SoftDropPuppi")[1].id() == kSubjetsPID);
  CHECK(back[0].subjets("SoftDropPuppi")[1].key() == 1);
  CHECK(back[1].nSubjetCollections() == 0);
  // user data
  CHECK(back[0].userFloat("NjettinessAK8Puppi:tau1") == 0.31f);
  CHECK(back[0].userInt("hfcentralEtaStripSize") == 2);
  CHECK_FALSE(back[1].hasUserFloat("NjettinessAK8Puppi:tau1"));
  // neutral values still neutral after I/O
  CHECK(back[0].jecFactor("Uncorrected") == 1.f);
  CHECK(back[1].groomedMass("SoftDropPuppi") == -1.0);
}

TEST_CASE("class version: a file written by this build is read back", "[GenJet][io][version]") {
  TClass* cl = TClass::GetClass("pat::io_v1::GenJet");
  REQUIRE(cl);
  CHECK(cl->GetClassVersion() == 3);
  {
    std::vector<pat::GenJet> subjetStore;
    writeCollection(kVersionFile, makeCollectionForIO(subjetStore));
  }
  // reopen: the streamer info stored in the file must be version 3 with this build's checksum
  {
    std::unique_ptr<TFile> f(TFile::Open(kVersionFile, "READ"));
    REQUIRE(f);
    std::unique_ptr<TList> infos(f->GetStreamerInfoList());
    REQUIRE(infos);
    TStreamerInfo* si = dynamic_cast<TStreamerInfo*>(infos->FindObject("pat::io_v1::GenJet"));
    REQUIRE(si);
    CHECK(si->GetClassVersion() == 3);
    CHECK(si->GetCheckSum() == cl->GetCheckSum());
  }
  const auto back = readCollection(kVersionFile);
  REQUIRE(back.size() == 2);
  CHECK(back[0].pt() == Approx(120.));
  CHECK(back[0].subjets("SoftDropPuppi").size() == 2);
  CHECK(back[0].genJetFwdRef().backRef().key() == 3);
}

// ==================================================================================================
// Further behaviour checks: constructors, generator constituents, zero-energy jet
// ==================================================================================================

TEST_CASE("RefToBase constructors (reco::GenJet and pat::GenJet)", "[GenJet][construction]") {
  const reco::GenJet rj = makeRecoGenJet(42.);
  std::vector<reco::GenJet> rcoll{makeRecoGenJet(10.), rj};
  edm::OrphanHandle<std::vector<reco::GenJet>> rh(&rcoll, kGenJetsPID);
  edm::RefToBase<reco::GenJet> rref(edm::Ref<std::vector<reco::GenJet>>(rh, 1));
  pat::GenJet fromRef = rref;  // implicit
  CHECK(fromRef.p4() == rj.p4());
  CHECK(fromRef.emEnergy() == 30.f);  // Specific kept
  CHECK(fromRef.originalObjectRef().id() == kGenJetsPID);
  CHECK(fromRef.originalObjectRef().key() == 1);
  CHECK(fromRef.originalObject() == &rcoll[1]);

  pat::GenJet pj(rj);
  pj.addUserFloat("x", 2.5f);
  std::vector<pat::GenJet> pcoll{pat::GenJet(makeRecoGenJet(11.)), pj};
  edm::OrphanHandle<std::vector<pat::GenJet>> ph(&pcoll, kSubjetsPID);
  edm::RefToBase<pat::GenJet> pref(edm::Ref<std::vector<pat::GenJet>>(ph, 1));
  pat::GenJet fromPatRef = pref;
  CHECK(fromPatRef.p4() == rj.p4());
  CHECK(fromPatRef.userFloat("x") == 2.5f);  // pat content copied
  // as pat::Jet(RefToBase<pat::Jet>): the original object becomes the pat jet it was made from
  CHECK(fromPatRef.originalObjectRef().id() == kSubjetsPID);
  CHECK(fromPatRef.originalObjectRef().key() == 1);
}

TEST_CASE("2.3 getGenConstituent(s): inherited, throws on packed constituents", "[GenJet][2.3][constituents]") {
  // kept as in reco::GenJet; works on GenParticle constituents, throws on packed ones
  std::vector<pat::PackedGenParticle> packed;
  for (auto const& gp : constituents())
    packed.emplace_back(gp, reco::GenParticleRef());  // 1-arg ctor reads motherRef(0)
  const edm::ProductID packedPID(1, 10);
  reco::GenJet rj(constituents()[0].p4(), reco::Particle::Point(0, 0, 0), makeSpecific());
  for (size_t i = 0; i < packed.size(); ++i)
    rj.addDaughter(reco::CandidatePtr(packedPID, &packed[i], i));
  pat::GenJet j(rj);
  REQUIRE(j.numberOfDaughters() == packed.size());
  CHECK(j.daughterPtr(2).key() == 2);  // the documented replacement
  const pat::GenJet& cj = j;           // the non-const daughter(i) of reco candidates returns nullptr
  CHECK(cj.daughter(2)->pt() == Approx(packed[2].pt()));
  CHECK_THROWS_WITH(j.getGenConstituents(), Catch::Matchers::ContainsSubstring("Invalid Constituent"));
  CHECK_THROWS_AS(j.getGenConstituent(0), cms::Exception);
  CHECK(j.getGenConstituent(99) == nullptr);  // out of range: nullptr (reco::GenJet)
  // GenParticle constituents: works
  pat::GenJet ok(makeRecoGenJet());
  CHECK(ok.getGenConstituent(1) == &constituents()[1]);
}

TEST_CASE("zero-energy jet: what each fraction returns", "[GenJet][2.3][zeroE]") {
  // A jet with E = 0 and an all-zero Specific (no generator jet should have it).
  // PF-named fractions keep pat::Jet's unguarded body x/energy(): 0/0 = NaN (no new guard, as decided).
  reco::GenJet rj(reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0, 0, 0), reco::GenJet::Specific());
  pat::GenJet j(rj);
  REQUIRE(j.energy() == 0.);
  CHECK(std::isnan(j.chargedHadronEnergyFraction()));
  CHECK(std::isnan(j.neutralHadronEnergyFraction()));
  CHECK(std::isnan(j.chargedEmEnergyFraction()));
  CHECK(std::isnan(j.neutralEmEnergyFraction()));
  CHECK(std::isnan(j.muonEnergyFraction()));
  CHECK(std::isnan(j.photonEnergyFraction()));
  CHECK(std::isnan(j.electronEnergyFraction()));
  CHECK(std::isnan(j.chargedMuEnergyFraction()));
  CHECK(j.jecFactor(0) == 1.f);
  CHECK(j.groomedMass() == -1.0);
}
