// -*- C++ -*-
//
// Package:     DataFormats/L1TCorrelator
// Class  :     TkBsCandidate
//

#include "DataFormats/L1TCorrelator/interface/TkPhiCandidate.h"
#include "DataFormats/L1TCorrelator/interface/TkBsCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <cmath>

using namespace l1t;

TkBsCandidate::TkBsCandidate() {}
TkBsCandidate::TkBsCandidate(const LorentzVector& p4, TkPhiCandidate cand1, TkPhiCandidate cand2)
    : L1Candidate(p4), phiCandList_{cand1, cand2} {}

// deltaR between the Phi pair
double TkBsCandidate::dRPhiPair() const {
  const LorentzVector& lva = phiCandidate(0).p4();
  const LorentzVector& lvb = phiCandidate(1).p4();
  return reco::deltaR(lva, lvb);
}
// position difference between track pair
double TkBsCandidate::dxyPhiPair() const {
  const TkPhiCandidate& phia = phiCandidate(0);
  const TkPhiCandidate& phib = phiCandidate(1);
  return std::sqrt(std::pow(phia.vx() - phib.vx(), 2) + std::pow(phia.vy() - phib.vy(), 2));
}
double TkBsCandidate::dzPhiPair() const { return (phiCandidate(0).vz() - phiCandidate(1).vz()); }
