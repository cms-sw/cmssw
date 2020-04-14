// -*- C++ -*-
//
// Package:     DataFormats/L1TCorrelator
// Class  :     TkPhiCandidate
//
#include "DataFormats/L1TCorrelator/interface/TkPhiCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <cmath>

using namespace l1t;

TkPhiCandidate::TkPhiCandidate() {}
TkPhiCandidate::TkPhiCandidate(const LorentzVector& p4,
                               const edm::Ptr<L1TTTrackType>& trkPtr1,
                               const edm::Ptr<L1TTTrackType>& trkPtr2)
    : L1Candidate(p4) {
  trkPtrList_.push_back(trkPtr1);
  trkPtrList_.push_back(trkPtr2);
}
// deltaR between track pair
double TkPhiCandidate::dRTrkPair() const {
  const edm::Ptr<L1TTTrackType>& itrk = trkPtr(0);
  const edm::Ptr<L1TTTrackType>& jtrk = trkPtr(1);

  math::PtEtaPhiMLorentzVector itrkP4(itrk->momentum().perp(), itrk->momentum().eta(), itrk->momentum().phi(), kmass);
  math::PtEtaPhiMLorentzVector jtrkP4(jtrk->momentum().perp(), jtrk->momentum().eta(), jtrk->momentum().phi(), kmass);
  return reco::deltaR(itrkP4, jtrkP4);
}

// difference from nominal mass
double TkPhiCandidate::dmass() const { return std::abs(phi_polemass - mass()); }
// position difference between track pair
double TkPhiCandidate::dxyTrkPair() const {
  const edm::Ptr<L1TTTrackType>& itrk = trkPtr(0);
  const edm::Ptr<L1TTTrackType>& jtrk = trkPtr(1);

  return std::sqrt(std::pow(itrk->POCA().x() - jtrk->POCA().x(), 2) + std::pow(itrk->POCA().y() - jtrk->POCA().y(), 2));
}
double TkPhiCandidate::dzTrkPair() const { return (trkPtr(0)->POCA().z() - trkPtr(1)->POCA().z()); }
double TkPhiCandidate::vx() const { return 0.5 * (trkPtr(0)->POCA().x() + trkPtr(1)->POCA().x()); }
double TkPhiCandidate::vy() const { return 0.5 * (trkPtr(0)->POCA().y() + trkPtr(1)->POCA().y()); }
double TkPhiCandidate::vz() const { return 0.5 * (trkPtr(0)->POCA().z() + trkPtr(1)->POCA().z()); }
