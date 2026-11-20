#ifndef PhysicsTools_PatAlgos_test_PatGenJetTestCommon_h
#define PhysicsTools_PatAlgos_test_PatGenJetTestCommon_h
// Shared pieces of the TestProcessor tests.

#include "DataFormats/PatCandidates/interface/GenJet.h"

#include <cmath>
#include <string>
#include <vector>

namespace patgenjettest {

  // The input module (test/PatGenJetTestInputs.cc). Its label ends in "ForPatGenJets", so that
  // labelPostfixesToStrip can be tested.
  inline std::string header(bool sortJets = false, bool unmappedConstituent = false, bool dropPartners = false) {
    return std::string(R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.genInputsForPatGenJets = cms.EDProducer("PatGenJetTestInputs", sortJets = cms.untracked.bool()_") +
           (sortJets ? "True" : "False") + "), unmappedConstituent = cms.untracked.bool(" +
           (unmappedConstituent ? "True" : "False") + "), dropPartners = cms.untracked.bool(" +
           (dropPartners ? "True" : "False") + "))\n";
  }

  // userData block with every PSet that PATUserDataHelper::fillDescription requires
  inline std::string userData(std::string const& floats, std::string const& ints, std::string const& strip) {
    return "cms.PSet(userFloats = cms.PSet(src = cms.VInputTag(" + floats + "), labelPostfixesToStrip = cms.vstring(" +
           strip + ")),\n" + "  userInts = cms.PSet(src = cms.VInputTag(" + ints +
           "), labelPostfixesToStrip = cms.vstring(" + strip + ")),\n" +
           "  userClasses = cms.PSet(src = cms.VInputTag()), userCands = cms.PSet(src = cms.VInputTag()),\n"
           "  userFunctions = cms.vstring(), userFunctionLabels = cms.vstring())";
  }

  inline std::vector<double> pts(std::vector<pat::GenJet> const& v) {
    std::vector<double> r;
    for (auto const& j : v)
      r.push_back(j.pt());
    return r;
  }
  template <typename C>
  inline std::vector<double> ptsOf(C const& v) {
    std::vector<double> r;
    for (auto const& j : v)
      r.push_back(j.pt());
    return r;
  }

  // genJetFwdRef of every entry: not null, the back ref is the reco::GenJet the entry was made from (same
  // product and key as originalObjectRef(), same pt). Returns the number of entries that do NOT satisfy this;
  // .get() / operator* resolve through the event.
  inline unsigned badBackRefs(std::vector<pat::GenJet> const& v) {
    unsigned bad = 0;
    for (auto const& j : v) {
      auto const& l = j.genJetFwdRef();
      if (l.isNull() || l.backRef().isNull() || l.backRef().id() != j.originalObjectRef().id() ||
          l.backRef().key() != j.originalObjectRef().key() || std::abs(l.backRef()->pt() - j.pt()) > 1e-5)
        ++bad;
    }
    return bad;
  }

  // The maps of the inputs (PatGenJetTestInputs.cc), as expected values.
  // genJetForwardMap: pt of the input jet -> key in "fwdjets" (their pt: 50, 30, 80, 12), -1 = no partner (dropPartners)
  inline int expectedJetFwdKey(double pt, bool dropPartners = false) {
    const int pt0 = static_cast<int>(std::lround(pt));
    return pt0 == 80 ? 2 : pt0 == 50 ? 0 : pt0 == 30 ? 1 : dropPartners ? -1 : 3;
  }
  // keys of the entries of a list in the product that is made of the list (then, if full, of all other numbers < n)
  inline std::vector<int> expectedKeys(std::vector<unsigned> const& list, unsigned n, bool full) {
    std::vector<int> key(n, -1);
    int next = 0;
    for (unsigned g : list)
      key[g] = next++;
    if (full)
      for (unsigned k = 0; k < n; ++k)
        if (key[k] < 0)
          key[k] = next++;
    return key;
  }
  // gen particle k -> key in "stage1" (map1) and in "stage2" (map1 then map2), -1 = no partner; nGen = 10 or 11 gens
  inline int expectedStage1Key(unsigned k, unsigned nGen, bool dropPartners = false) {
    return expectedKeys({1, 5, 8, 3, 0, 7, 2}, nGen, !dropPartners)[k];
  }
  inline int expectedStage2Key(unsigned k, unsigned nGen, bool dropPartners = false) {
    const int s1 = expectedStage1Key(k, nGen, dropPartners);
    if (s1 < 0)
      return -1;
    const unsigned n1 = dropPartners ? 7 : nGen;
    return expectedKeys({0, 1, 3, 4, 5}, n1, !dropPartners)[s1];
  }
  // pt of the gen particles (the copies in the stages have the same pt)
  constexpr double kGenPtExpected[10] = {5., 40., 12., 25., 8., 30., 3., 18., 22., 6.};

}  // namespace patgenjettest

#endif
