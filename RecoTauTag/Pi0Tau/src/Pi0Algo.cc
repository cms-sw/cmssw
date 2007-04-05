// -*- C++ -*-
//
// Package:    Pi0Algo
// Class:      Pi0Algo
// 
/**\class Pi0Algo Pi0Algo.cc RecoTauTag/Pi0Tau/src/Pi0Algo.cc

 Description: Make Pi0 from either PFCandidate or BasicCluster

 Implementation:

       1. currently selecting only photons from PFCandidate by checking their charge

*/
//
// Original Author:  Dongwook Jang
//         Created:  Tue Jan  9 16:40:36 CST 2007
// $Id: Pi0Algo.cc,v 1.1 2007/03/27 21:32:04 dwjang Exp $
//
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoTauTag/Pi0Tau/interface/Pi0Algo.h"

#include "Math/GenVector/VectorUtil.h"

#include <iostream>
#include <algorithm>
#include <cmath>


using namespace reco;

//
// constructors and destructor
//
Pi0Algo::Pi0Algo(reco::TrackRef seedTrack){
  coneSize_  = 0.524; // corresponding to 30 degree if 3D angle used
  use3DAngle_ = false;
  ecalEntrance_ = 129.0;
  massRes_ = 0.03;
  seedTrack_ = seedTrack;
  pi0Collection_.clear();
}


Pi0Algo::~Pi0Algo()
{
  pi0Collection_.clear();
}


void Pi0Algo::fillPi0sUsingPF(edm::Handle<reco::PFCandidateCollection> &pFCandidateHandle){

  LogDebug("Pi0Algo") << "The validity of seedTrack is " << (seedTrack_.isNull() ? 0 : 1) << "\n";

  if(seedTrack_.isNull()){
    LogDebug("Pi0Algo") << "seedTrack is not valid";
    return;
  }

  LogDebug("Pi0Algo") << "vertex of seedTrack : ("
		      << seedTrack_->vertex().x() << ","
		      << seedTrack_->vertex().y() << ","
		      << seedTrack_->vertex().z() << ")\n";
  
  if(!pFCandidateHandle.isValid()) {
    LogDebug("Pi0Algo") << "PFCandidateCollection is not valid";
    return;
  }

  const reco::PFCandidateCollection& pFCandidateCollection = *(pFCandidateHandle.product());

  reco::PFCandidateRefVector photonCands;
  int icand = 0;
  for(reco::PFCandidateCollection::const_iterator  ic = pFCandidateCollection.begin();
       ic != pFCandidateCollection.end(); ++ic) {
    const reco::PFCandidate *cand = &(*ic);
    reco::PFCandidateRef candRef(pFCandidateHandle,icand);
    icand++;
    if(cand->particleId() != reco::PFCandidate::gamma) continue;

    double dist = use3DAngle_ ? ROOT::Math::VectorUtil::Angle(seedTrack_->momentum(),cand->momentum()) :
      ROOT::Math::VectorUtil::DeltaR(seedTrack_->momentum(),cand->momentum());

    if(dist > coneSize_) continue;
    photonCands.push_back(candRef);
  }

  LogDebug("Pi0Algo") << " size of photonCandidates : " << photonCands.size() << "\n";

  reco::PFCandidateRefVector usedPhotons;
  int nPhotons = photonCands.size();
  for(reco::PFCandidateRefVector::const_iterator ip = photonCands.begin();
      ip != photonCands.end(); ++ip){
    const reco::PFCandidateRef candRef = *ip;
    reco::PFCandidateRefVector pi0Cands;
    if(binary_search(usedPhotons.begin(),usedPhotons.end(),candRef)) continue;
    else {
      usedPhotons.push_back(candRef);
      pi0Cands.push_back(candRef);
    }

    int type = reco::Pi0::UnResolved;

    math::XYZPoint pos = calculatePositionAtEcal(candRef->p4());
    math::XYZTLorentzVector p4 = calculateMomentumWRT(candRef->p4(),seedTrack_->vertex());

    LogDebug("Pi0Algo") << "Compare original and calculated photon momentum and position ------------\n"
			<< " The position at ecal : ("
			<< pos.x() << ","
			<< pos.y() << ","
			<< pos.z() << ")\n"
			<< " The original  momentum : (" 
			<< candRef->p4().px() << ","
			<< candRef->p4().py() << ","
			<< candRef->p4().pz() << ","
			<< candRef->p4().energy() << ")\n"
			<< " The corrected momentum : ("
			<< p4.px() << ","
			<< p4.py() << ","
			<< p4.pz() << ","
			<< p4.energy() << ")\n";

    // Now find a pair of photons which have a minimum invariant mass.
    // The second photon shouldn't be used previously, say, not an element of usedPhotons.

    double minMass = 999.0;
    reco::PFCandidateRef pho2;
    if((nPhotons > 1) && (ip != photonCands.end())){
      for(reco::PFCandidateRefVector::const_iterator ip2 = ip+1;
	  ip2 != photonCands.end(); ++ip2){
	reco::PFCandidateRef candRef2 = *ip2;
	if(binary_search(usedPhotons.begin(),usedPhotons.end(),candRef2)) continue;
	math::XYZTLorentzVector sum = p4 + calculateMomentumWRT(candRef2->p4(),seedTrack_->vertex());
	double massDiff = std::abs(sum.M()-PI0MASS);
	if(massDiff < minMass){
	  minMass = massDiff;
	  pho2 = candRef2;
	}
      }
    }

    LogDebug("Pi0Algo") << "minMass of two photons : " << minMass << "\n";

    if(pho2.isNull()){
      reco::Pi0 pi0(type,p4.energy(),pos,p4,pi0Cands);
      pi0Collection_.push_back(pi0);
    }
    else if(minMass < massRes_){
      usedPhotons.push_back(pho2);
      pi0Cands.push_back(pho2);
      math::XYZPoint pos2 = calculatePositionAtEcal(pho2->p4());
      math::XYZPoint avPos(0.5*(pos.x()+pos2.x()),0.5*(pos.y()+pos2.y()),0.5*(pos.z()+pos2.z()));
      math::XYZTLorentzVector sumP4 = p4 + calculateMomentumWRT(pho2->p4(),seedTrack_->vertex());
      type = reco::Pi0::Resolved;
      reco::Pi0 pi0(type,sumP4.energy(),avPos,sumP4,pi0Cands);
      pi0Collection_.push_back(pi0);

      LogDebug("Pi0Algo") << "Found resolved photons -----------------------\n"
			  << "pos   : (" << pos.x() << "," << pos.y() << "," << pos.z() << ")\n"
			  << "pos2  : (" << pos2.x() << "," << pos2.y() << "," << pos2.z() << ")\n"
			  << "avPos : (" << avPos.x() << "," << avPos.y() << "," << avPos.z() << ")\n"
			  << "sumP4 : (" << sumP4.px() << "," << sumP4.py() << "," << sumP4.pz() << "," << sumP4.energy() << ")\n";

    }// else if

    pi0Cands.clear();
  }// for cluster


  // clear temporary vector's
  photonCands.clear();
  usedPhotons.clear();

}


math::XYZPoint Pi0Algo::calculatePositionAtEcal(const math::XYZTLorentzVector &momentum) const {

  // This is also temporary.
  // Position at ECAL can be obtained from ecal cluster links from candidates
  // when PFCandidate object is made.

  double phi = momentum.phi();
  double theta = momentum.theta();

  // ecalEntrance will be replaced by the official geometry constant later.
  // This works only for barrels, but I think it should work for endcap as well because
  // this is just a reference point to recalculate momentum w.r.t a given vertex.

  math::XYZPoint pos(ecalEntrance_*std::cos(phi),ecalEntrance_*std::sin(phi),ecalEntrance_/std::tan(theta));

  return pos;
}


math::XYZTLorentzVector Pi0Algo::calculateMomentumWRT(const math::XYZTLorentzVector &momentum, const math::XYZPoint &vertex) const {

  math::XYZVector dir = calculatePositionAtEcal(momentum) - vertex;
  math::XYZVector pos = dir.unit() * momentum.energy();

  math::XYZTLorentzVector p4(pos.x(),pos.y(),pos.z(),momentum.energy());

  return p4;
}
