// -*- C++ -*-
//
// Package:    Tau3DAlgo
// Class:      Tau3DAlgo
// 
/**\class Tau3DAlgo Tau3DAlgo.cc RecoParticleFlowTau/Tau3DAlgo/src/Tau3DAlgo.cc

 Description: Make Tau3D from PFPi0 and Track

 Implementation:

        1. select seedTracks from TrackCollection
        2. associate tracks and pi0s in 30 degree around a seedTrack
 
*/
//
// Original Author:  Dongwook Jang
//         Created:  Tue Jan  9 16:40:36 CST 2007
// $Id$
//
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "RecoTauTag/Pi0Tau/interface/Pi0Algo.h"
#include "RecoTauTag/Pi0Tau/interface/Tau3DAlgo.h"

using namespace reco;

bool trackPtGreaterThan(const reco::TrackRef &left, const reco::TrackRef &right){
  return (left->momentum().Rho() > right->momentum().Rho());
}


//
// constructors and destructor
//
Tau3DAlgo::Tau3DAlgo(edm::Handle<reco::TrackCollection> *trackHandle) {

  trackHandle_ = trackHandle;

// threshold for seed track
  seedTrackPtThreshold_ = 5.0;

// radian for 30 degree : 30.0/180.0 * TMath::Pi() = 0.5236
  tauOuterConeSize_     = 0.5236;

}


Tau3DAlgo::~Tau3DAlgo()
{

  trackHandle_ = 0;

  trackRefs_.clear();
  seedTrackRefs_.clear();

}


void Tau3DAlgo::fillTau3Ds(edm::Handle<reco::PFCandidateCollection> &pFCandidateHandle){

  fillRefVectors();

  LogDebug("Tau3DAlgo::fillTau3Ds") << " seedTrackCandidateRefs_.size() : " << seedTrackCandidateRefs_.size() << "\n";

  // sort seed track candidate collection in pt descending order
  std::sort(seedTrackCandidateRefs_.begin(),seedTrackCandidateRefs_.end(),trackPtGreaterThan);

  findSeedTracks();

  LogDebug("Tau3DAlgo::fillTau3Ds") << " seedTrackRefs_.size() : " << seedTrackRefs_.size() << "\n";

  for(std::vector<reco::TrackRef>::const_iterator sIter = seedTrackRefs_.begin();
      sIter != seedTrackRefs_.end(); sIter++){

    const reco::TrackRef seedTrk = *sIter;

    if(seedTrk.isNull()) continue;

    reco::TrackRefVector trackColl;

    for(std::vector<reco::TrackRef>::const_iterator tIter = trackRefs_.begin();
      tIter != trackRefs_.end(); tIter++){
      const reco::TrackRef trkRef = *tIter;
      double angle = ROOT::Math::VectorUtil::Angle(seedTrk->momentum(),trkRef->momentum());
      if(angle < tauOuterConeSize_) trackColl.push_back(trkRef);
    }

    reco::Pi0Algo pi0Algo(seedTrk);
    pi0Algo.fillPi0sUsingPF(pFCandidateHandle,tauOuterConeSize_);

    reco::Tau3D tau3D(seedTrk,trackColl,pi0Algo.pi0Collection());
    tau3DCollection_.push_back(tau3D);

    trackColl.clear();

  }// for seed track iter



}



void Tau3DAlgo::fillRefVectors(){

  trackRefs_.clear();
  seedTrackCandidateRefs_.clear();

  if(trackHandle_->isValid()) {
    const reco::TrackCollection trackCollection = *(trackHandle_->product());

    // Let's make a collection of track references just once because this will be used later on
    int itrk=0;
    for(reco::TrackCollection::const_iterator tIter = trackCollection.begin();
	tIter != trackCollection.end(); tIter++){
      const reco::TrackRef trkRef(*trackHandle_,itrk);
      trackRefs_.push_back(trkRef);
      if(trkRef->momentum().Rho() > seedTrackPtThreshold_) seedTrackCandidateRefs_.push_back(trkRef);
      itrk++;
    }
  }
  else {
    LogDebug("Tau3DAlgo") << "TrackCollection is not valid\n";
  }

}



void Tau3DAlgo::findSeedTracks(){

  // now find seed tracks above a certain threshold which should be the highest one in 30 degree cone

  seedTrackRefs_.clear();

  for(std::vector<reco::TrackRef>::const_iterator tIter = seedTrackCandidateRefs_.begin();
      tIter != seedTrackCandidateRefs_.end(); tIter++){
    const reco::TrackRef trkRef = *tIter;
    double pt = trkRef->pt();

    bool anyTrackHigherThanThisIn30 = false;
    for(std::vector<reco::TrackRef>::const_iterator tIter2 = seedTrackCandidateRefs_.begin();
      tIter2 != seedTrackCandidateRefs_.end(); tIter2++){
      const reco::TrackRef trkRef2 = *tIter2;
      if(trkRef == trkRef2) continue;
      double pt2 = trkRef2->pt();
      if(pt2 < pt) continue;
      double angle = ROOT::Math::VectorUtil::Angle(trkRef->momentum(),trkRef2->momentum());
      if(angle > tauOuterConeSize_) continue;
      anyTrackHigherThanThisIn30 = true;
    }// for tIter2

    if(anyTrackHigherThanThisIn30) continue;
    if(binary_search(seedTrackRefs_.begin(),seedTrackRefs_.end(),trkRef)) continue;
    seedTrackRefs_.push_back(trkRef);

  }// for tIter

}

