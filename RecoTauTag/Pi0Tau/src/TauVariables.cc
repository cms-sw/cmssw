// -*- C++ -*-
//
// Package:    TauVariables
// Class:      TauVariables
// 
/**\class TauVariables TauVariables.cc RecoTauTag/Pi0Tau/src/TauVariables.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dongwook Jang
//         Created:  Tue Jan  9 16:40:36 CST 2007
// $Id$
//
//

#include "Math/GenVector/VectorUtil.h"
#include "RecoTauTag/Pi0Tau/interface/TauVariables.h"

#include <cmath>

const double PIMASS = 0.139;

using namespace reco;

//
// constructors and destructor
//

TauVariables::TauVariables(const reco::Tau3D *tau, const edm::Handle<reco::IsolatedTauTagInfoCollection> *tauTagInfoHandle, int algorithm)
{

  this->init();

  tau3D_ = tau;
  tauTagInfoHandle_ = tauTagInfoHandle;
  algorithm_ = algorithm;

  dZTrackAssociation_ = 5.0; // can be changed by setters after calling constructors

  //  this->makeVariables();

}


TauVariables::~TauVariables()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void TauVariables::init(){

  tracksMomentum_ *= 0;
  pi0sMomentum_ *= 0;

  seedTrack_ = TrackRef();
  algorithm_ = 0;
  signalConeSize_ = 0.0;
  isolationConeSize_ = 0.0;
  nSignalTracks_ = 0;
  nSignalPi0s_ = 0;
  nIsolationTracks_ = 0;
  nIsolationPi0s_ = 0;
  signalCharge_ = 0;
  maxPtTrackInIsolation_ = reco::TrackRef();
  maxPtPi0InIsolation_ = reco::Pi0();

  signalTracks_.clear();
  signalPi0s_.clear();
  isolationTracks_.clear();
  isolationPi0s_.clear();
  tauTagRef_ = IsolatedTauTagInfoRef();
  tau3D_ = 0;
  tauTagInfoHandle_ = 0;

}


void TauVariables::makeVariables(){

  seedTrack_ = tau3D().seedTrack();
  if(seedTrack_.isNull()) return;

  const IsolatedTauTagInfoCollection &tauTagColl = *(tauTagInfoHandle_->product());
  IsolatedTauTagInfoCollection::const_iterator endIter = tauTagColl.end();
  IsolatedTauTagInfoCollection::const_iterator tauIter = tauTagColl.begin();
  int itau=0;
  double minAlpha = 999.0;
  for(; tauIter != endIter; tauIter++){
    const IsolatedTauTagInfoRef tauTagRef(*tauTagInfoHandle_,itau);
    itau++;
    const TrackRef seed = tauIter->leadingSignalTrack(0.524,5.0);
    if(seed.isNull()) continue;
    double alpha = ROOT::Math::VectorUtil::Angle(seedTrack_->momentum(),seed->momentum());
    if(alpha < minAlpha){
      minAlpha = alpha;
      tauTagRef_ = tauTagRef;
    }
  }


  signalConeSize_    = 0.175;
  isolationConeSize_ = 0.524;

  if(algorithm_ == 1){
    signalConeSize_    = 0.175;
    if(!tauTagRef_.isNull()) signalConeSize_ = min(0.175, max(5.0/tauTagRef_->jet().energy(), 0.05));
    isolationConeSize_ = 0.524;
  }


  double maxPtIso = 0.0;
  maxPtTrackInIsolation_ = reco::TrackRef();

  for(TrackRefVector::const_iterator trkIter = tau3D().tracks().begin();
      trkIter != tau3D().tracks().end(); trkIter++){
    const TrackRef trk = *trkIter;

    double dZ = seedTrack_->vertex().z() - trk->vertex().z();
    if(std::abs(dZ) > dZTrackAssociation_) continue;

    double energy = std::sqrt(trk->momentum().Mag2() + PIMASS*PIMASS);
    math::XYZTLorentzVector trkP4(trk->momentum().X(),trk->momentum().Y(),trk->momentum().Z(),energy);

    double pt = trk->pt();
    if(pt < 1.0) continue;

    double angle = ROOT::Math::VectorUtil::Angle(seedTrack_->momentum(),trk->momentum());

    if(angle > isolationConeSize_) continue;

    if(angle < signalConeSize_){
      nSignalTracks_++;
      signalCharge_ += trk->charge();
      tracksMomentum_ += trkP4;
      signalTracks_.push_back(trk);
    }
    else {
      nIsolationTracks_++;
      isolationTracks_.push_back(trk);
      if(pt > maxPtIso){
	maxPtIso = pt;
	maxPtTrackInIsolation_ = trk;
      }
    }

  }//trkIter


  maxPtIso = 0.0;
  maxPtPi0InIsolation_ = reco::Pi0();

  for(Pi0Collection::const_iterator pIter = tau3D().pi0s().begin(); pIter != tau3D().pi0s().end(); pIter++){
    const Pi0 &pi0 = *pIter;

    math::XYZTLorentzVector p4 = pi0.momentum(seedTrack_->vertex());

    double et = p4.Et();
    if(et < 0.1) continue;

    double angle = ROOT::Math::VectorUtil::Angle(seedTrack_->momentum(),p4);

    if(angle > isolationConeSize_) continue;

    if(angle < signalConeSize_){
      nSignalPi0s_++;
      pi0sMomentum_ += p4;
      signalPi0s_.push_back(pi0);
    }
    else {
      nIsolationPi0s_++;
      isolationPi0s_.push_back(pi0);
      if(et > maxPtIso){
	maxPtIso = et;
	maxPtPi0InIsolation_ = pi0;
      }
    }
  }// for 


}
