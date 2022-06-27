/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusVertex.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

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
BPHPlusMinusVertex::BPHPlusMinusVertex(const BPHEventSetupWrapper* es)
    : BPHDecayVertex(es), oldA(true), inRPhi(nullptr) {}

//--------------
// Destructor --
//--------------
BPHPlusMinusVertex::~BPHPlusMinusVertex() { delete inRPhi; }

//--------------
// Operations --
//--------------
const ClosestApproachInRPhi& BPHPlusMinusVertex::cAppInRPhi() const {
  if (oldA)
    computeApp();
  if (inRPhi == nullptr) {
    static const ClosestApproachInRPhi ca;
    return ca;
  }
  return *inRPhi;
}

bool BPHPlusMinusVertex::chkSize(const string& msg) const { return chkSize(daughters(), msg); }

void BPHPlusMinusVertex::setNotUpdated() const {
  BPHDecayVertex::setNotUpdated();
  oldA = true;
  return;
}

void BPHPlusMinusVertex::computeApp() const {
  static const string msg = "BPHPlusMinusVertex::computeApp: incomplete, no closest approach available";
  delete inRPhi;
  if (!chkSize(msg)) {
    inRPhi = nullptr;
    return;
  }
  inRPhi = new ClosestApproachInRPhi;
  const vector<reco::TransientTrack>& ttk = transientTracks();
  const reco::TransientTrack& ttp = ttk[0];
  const reco::TransientTrack& ttn = ttk[1];
  inRPhi->calculate(ttp.impactPointTSCP().theState(), ttn.impactPointTSCP().theState());
  oldA = false;
  return;
}
