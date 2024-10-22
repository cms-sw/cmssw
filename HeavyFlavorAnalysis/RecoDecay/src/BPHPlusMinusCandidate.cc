/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAddFourMomenta.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//---------------
// C++ Headers --
//---------------

using namespace std;

//-------------------
// Initializations --
//-------------------

//----------------
// Constructors --
//----------------
BPHPlusMinusCandidate::BPHPlusMinusCandidate(const BPHEventSetupWrapper* es)
    : BPHDecayVertex(es), BPHPlusMinusVertex(es), BPHRecoCandidate(es) {}

BPHPlusMinusCandidate::BPHPlusMinusCandidate(const BPHEventSetupWrapper* es,
                                             const BPHRecoBuilder::ComponentSet& compList)
    : BPHDecayMomentum(compList.daugMap, compList.compMap),
      BPHDecayVertex(this, es),
      BPHKinematicFit(this),
      BPHPlusMinusVertex(es),
      BPHRecoCandidate(es, compList) {}

//--------------
// Operations --
//--------------
void BPHPlusMinusCandidate::add(const string& name, const reco::Candidate* daug, double mass, double sigma) {
  add(name, daug, "cfhpmig", mass, sigma);
  return;
}

void BPHPlusMinusCandidate::add(
    const string& name, const reco::Candidate* daug, const string& searchList, double mass, double sigma) {
  const vector<const reco::Candidate*>& dL = daughters();
  bool accept = false;
  switch (dL.size()) {
    case 0:
      accept = true;
      break;
    case 1:
      if ((daug->charge() * dL.front()->charge()) > 0) {
        edm::LogPrint("TooManyParticles") << "BPHPlusMinusCandidate::add: "
                                          << "already containing same sign particle, add rejected";
        return;
      }
      accept = true;
      break;
    default:
      edm::LogPrint("TooManyParticles") << "BPHPlusMinusCandidate::add: "
                                        << "complete, add rejected";
      return;
  }
  if (accept)
    addK(name, daug, searchList, mass, sigma);
  return;
}

vector<BPHPlusMinusConstCandPtr> BPHPlusMinusCandidate::build(
    const BPHRecoBuilder& builder, const string& nPos, const string& nNeg, double mass, double msig) {
  vector<BPHPlusMinusConstCandPtr> cList;
  class ChargeSelect : public BPHRecoSelect {
  public:
    ChargeSelect(int c) : charge(c) {}
    ~ChargeSelect() override = default;
    bool accept(const reco::Candidate& cand) const override { return ((charge * cand.charge()) > 0); }

  private:
    int charge;
  };
  ChargeSelect tkPos(+1);
  ChargeSelect tkNeg(-1);
  builder.filter(nPos, tkPos);
  builder.filter(nNeg, tkNeg);
  fill<BPHPlusMinusCandidate>(cList, builder, mass, msig);
  return cList;
}

/// clone object, cloning daughters as well up to required depth
/// level = -1 to clone all levels
BPHRecoCandidate* BPHPlusMinusCandidate::clone(int level) const {
  BPHPlusMinusCandidate* ptr = new BPHPlusMinusCandidate(getEventSetup());
  fill(ptr, level);
  return ptr;
}

const pat::CompositeCandidate& BPHPlusMinusCandidate::composite() const {
  static const pat::CompositeCandidate compCand;
  static const string msg = "BPHPlusMinusCandidate incomplete, no composite available";
  if (!chkSize(msg))
    return compCand;
  return BPHDecayMomentum::composite();
}

bool BPHPlusMinusCandidate::isCowboy() const {
  static const string msg = "BPHPlusMinusCandidate incomplete, no cowboy/sailor classification";
  return (chkSize(msg) && phiDiff());
}

bool BPHPlusMinusCandidate::isSailor() const {
  static const string msg = "BPHPlusMinusCandidate incomplete, no cowboy/sailor classification";
  return (chkSize(msg) && !phiDiff());
}

bool BPHPlusMinusCandidate::phiDiff() const {
  const vector<const reco::Candidate*>& dL = daughters();
  int idPos = (dL.front()->charge() > 0 ? 0 : 1);
  return reco::deltaPhi(dL[idPos]->phi(), dL[1 - idPos]->phi()) > 0;
}
