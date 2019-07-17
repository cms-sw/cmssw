// -*- C++ -*-
//
// Package:     DataFormats/L1TrackTrigger
// Class  :     L1TkBsCandidate
// 

#include "DataFormats/L1TrackTrigger/interface/L1TkPhiCandidate.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkBsCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace l1t;

L1TkBsCandidate::L1TkBsCandidate()
{
}
L1TkBsCandidate::L1TkBsCandidate(const LorentzVector& p4,
				 L1TkPhiCandidate cand1,
				 L1TkPhiCandidate cand2)
: L1Candidate(p4)
{
  phiCandList_.push_back(cand1);  
  phiCandList_.push_back(cand2);  
}
// deltaR between the Phi pair
double L1TkBsCandidate::dRPhiPair() const {
  const LorentzVector& lva = getPhiCandidate(0).p4();
  const LorentzVector& lvb = getPhiCandidate(1).p4();
  return reco::deltaR(lva, lvb);
}
// position difference between track pair
double L1TkBsCandidate::dxyPhiPair() const {
  const L1TkPhiCandidate& phia = getPhiCandidate(0); 
  const L1TkPhiCandidate& phib = getPhiCandidate(1); 
  return std::sqrt(std::pow(phia.vx() - phib.vx(), 2) 
                 + std::pow(phia.vy() - phib.vy(), 2));
}
double L1TkBsCandidate::dzPhiPair() const {
  return (getPhiCandidate(0).vz() - getPhiCandidate(1).vz());
}
