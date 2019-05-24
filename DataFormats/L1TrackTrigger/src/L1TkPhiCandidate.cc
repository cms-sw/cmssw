// -*- C++ -*-
//
// Package:     DataFormats/L1TrackTrigger
// Class  :     L1TkPhiCandidate
// 
#include "DataFormats/L1TrackTrigger/interface/L1TkPhiCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace l1t;

L1TkPhiCandidate::L1TkPhiCandidate()
{
}
L1TkPhiCandidate::L1TkPhiCandidate(const LorentzVector& p4,
				   const edm::Ptr<L1TTTrackType>& trkPtr1,
				   const edm::Ptr<L1TTTrackType>& trkPtr2)
: L1Candidate(p4)
{
  trkPtrList_.push_back(trkPtr1);  
  trkPtrList_.push_back(trkPtr2);  
}
// deltaR between track pair
double L1TkPhiCandidate::dRTrkPair() const {
  const edm::Ptr<L1TTTrackType>& itrk = getTrkPtr(0); 
  const edm::Ptr<L1TTTrackType>& jtrk = getTrkPtr(1); 

  math::PtEtaPhiMLorentzVector itrkP4(itrk->getMomentum().perp(), 
				      itrk->getMomentum().eta(), 
				      itrk->getMomentum().phi(), 
				      kmass);
  math::PtEtaPhiMLorentzVector jtrkP4(jtrk->getMomentum().perp(), 
				      jtrk->getMomentum().eta(), 
				      jtrk->getMomentum().phi(), 
				      kmass);
  return reco::deltaR(itrkP4, jtrkP4);
}

// difference from nominal mass
double L1TkPhiCandidate::dmass() const {
  return std::fabs(phi_polemass - mass());
}
// position difference between track pair
double L1TkPhiCandidate::dxyTrkPair() const {
  const edm::Ptr<L1TTTrackType>& itrk = getTrkPtr(0); 
  const edm::Ptr<L1TTTrackType>& jtrk = getTrkPtr(1); 

  return std::sqrt(std::pow(itrk->getPOCA(5).x() - jtrk->getPOCA(5).x(), 2) 
                 + std::pow(itrk->getPOCA(5).y() - jtrk->getPOCA(5).y(), 2));
}
double L1TkPhiCandidate::dzTrkPair() const {
  return (getTrkPtr(0)->getPOCA(5).z() - getTrkPtr(1)->getPOCA(5).z());
}
double L1TkPhiCandidate::vx() const {
  return 0.5 * (getTrkPtr(0)->getPOCA(5).x() + getTrkPtr(1)->getPOCA(5).x());
}
double L1TkPhiCandidate::vy() const {
  return 0.5 * (getTrkPtr(0)->getPOCA(5).y() + getTrkPtr(1)->getPOCA(5).y());
}
double L1TkPhiCandidate::vz() const {
  return 0.5 * (getTrkPtr(0)->getPOCA(5).z() + getTrkPtr(1)->getPOCA(5).z());
}
