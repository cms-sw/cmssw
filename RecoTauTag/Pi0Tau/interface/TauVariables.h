#ifndef RecoTauTag_Pi0Tau_TauVariables_h_
#define RecoTauTag_Pi0Tau_TauVariables_h_

// -*- C++ -*-
//
// Package:    TauVariables
// Class:      TauVariables
// 
/**\class TauVariables TauVariables.h RecoTauTag/Pi0Tau/interface/TauVariables.h

 Description: to calculate tau variables using Pi0 and tracks

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dongwook Jang
//         Created:  Tue Jan  9 16:40:36 CST 2007
// $Id$
//
//


// system include files
#include <memory>
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfoFwd.h"
#include "RecoTauTag/Pi0Tau/interface/Pi0.h"
#include "RecoTauTag/Pi0Tau/interface/Pi0Fwd.h"
#include "RecoTauTag/Pi0Tau/interface/Tau3D.h"
#include "RecoTauTag/Pi0Tau/interface/Tau3DFwd.h"

//
// class decleration
//

class TauVariables {

 public:

  TauVariables() { this->init(); }

  TauVariables(const reco::Tau3D* tau, const edm::Handle<reco::IsolatedTauTagInfoCollection> *tauTagInfoHandle, int algorithm=0);

  ~TauVariables();

  // \return tracks + pi0s momentum in the signal cone
  math::XYZTLorentzVector momentum() { return (tracksMomentum_ + pi0sMomentum_); }

  // \return tracks momentum in the signal cone
  math::XYZTLorentzVector tracksMomentum() const { return tracksMomentum_; }

  // \return pi0s momentum in the signal cone
  math::XYZTLorentzVector pi0sMomentum() const { return pi0sMomentum_; }

  // \return seed track
  reco::TrackRef seedTrack() const { return seedTrack_; }

  // \return algorithm
  int algorithm() const { return algorithm_; }

  // \return signal cone size
  double signalConeSize() const { return signalConeSize_; }

  // \return isolation cone size
  double isolationConeSize() const { return isolationConeSize_; }

  // \return number of tracks in signal cone
  int nSignalTracks() const { return nSignalTracks_; }

  // \return number of pi0s in signal cone
  int nSignalPi0s() const { return nSignalPi0s_; }

  // \return number of tracks in isolation cone
  int nIsolationTracks() const { return nIsolationTracks_; }

  // \return number of pi0s in isolation cone
  int nIsolationPi0s() const { return nIsolationPi0s_; }

  // \return sum charge of tracks in signal cone
  int signalCharge() const { return signalCharge_; }

  // \return mass of tracks and pi0s in signal cone
  double signalMass() { return momentum().M(); }

  // \return mass of tracks in signal cone
  double signalTracksMass() { return tracksMomentum().M(); }

  // \return maximum pt track in isolation cone
  reco::TrackRef const &maxPtTrackInIsolation() const { return maxPtTrackInIsolation_; }

  // \return maximum pt pi0 in isolation cone
  reco::Pi0 const &maxPtPi0InIsolation() const { return maxPtPi0InIsolation_; }

  // \return collection of tracks in signal cone
  reco::TrackRefVector const &signalTracks() const { return signalTracks_; }

  // \return collection of pi0s in signal cone
  reco::Pi0Collection const &signalPi0s() const { return signalPi0s_; }

  // \return collection of tracks in isolation cone
  reco::TrackRefVector const &isolationTracks() const { return isolationTracks_; }

  // \return collection of pi0s in isolation cone
  reco::Pi0Collection const &isolationPi0s() const { return isolationPi0s_; }

  // \return a reference to IsolatedTauTagInfoRef
  reco::IsolatedTauTagInfoRef tauTagRef() const { return tauTagRef_; }

  // \return a reference of Tau3D
  reco::Tau3D const &tau3D() { return *tau3D_; }

  // initialize members
  void init();

  // make tau variables (main algorithm)
  void makeVariables();

  // setters

  void setDZTrackAssociation(double v) { dZTrackAssociation_ = v; }

 private:

  // tracks momentum in the signal cone
  math::XYZTLorentzVector tracksMomentum_;

  // pi0s momentum in the signal cone
  math::XYZTLorentzVector pi0sMomentum_;

  // seed track ie. highest pt track
  reco::TrackRef seedTrack_;

  // algorithm represents signal cone definition
  // 0(default) : using fixed signal cone
  // 1          : using shrinking signal cone
  int algorithm_;

  // signal cone size
  double signalConeSize_;

  // isolation cone size
  double isolationConeSize_;

  // number of tracks in signal cone
  int nSignalTracks_;

  // number of pi0s in signal cone
  int nSignalPi0s_;

  // number of tracks in isolation cone
  int nIsolationTracks_;

  // number of pi0s in isolation cone
  int nIsolationPi0s_;

  // sum charge of tracks in signal cone
  int signalCharge_;

  // maximum pt track in isolation cone
  reco::TrackRef maxPtTrackInIsolation_;

  // maximum pt pi0 in isolation cone
  reco::Pi0 maxPtPi0InIsolation_;

  // collection of tracks in signal cone
  reco::TrackRefVector signalTracks_;

  // collection of pi0s in signal cone
  reco::Pi0Collection signalPi0s_;

  // collection of tracks in isolation cone
  reco::TrackRefVector isolationTracks_;

  // collection of pi0s in isolation cone
  reco::Pi0Collection isolationPi0s_;

  // a reference to IsolatedTauTagInfoRef
  reco::IsolatedTauTagInfoRef tauTagRef_;

  // a reference to Tau3D
  const reco::Tau3D *tau3D_;

  // a reference to IsolatedTauTagInfoCollection
  const edm::Handle<reco::IsolatedTauTagInfoCollection> *tauTagInfoHandle_;

  // parameters for algorithms

  double dZTrackAssociation_;

};
#endif

