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
// $Id: TauVariables.h,v 1.2 2007/04/05 19:27:49 dwjang Exp $
//
//


// system include files
#include <memory>
#include "DataFormats/Common/interface/Handle.h"
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

  TauVariables(const reco::Tau3D* tau, const edm::Handle<reco::IsolatedTauTagInfoCollection> *tauTagInfoHandle);

  ~TauVariables();

  // \return tracks + pi0s momentum in the signal cone
  math::XYZTLorentzVector momentum() { return (tracksMomentum_ + pi0sMomentum_); }

  // \return tracks momentum in the signal cone
  math::XYZTLorentzVector tracksMomentum() const { return tracksMomentum_; }

  // \return pi0s momentum in the signal cone
  math::XYZTLorentzVector pi0sMomentum() const { return pi0sMomentum_; }

  // \return seed track
  reco::TrackRef seedTrack() const { return seedTrack_; }

  // \return use3DAngle
  int use3DAngle() const { return use3DAngle_; }

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

  // make tau variables (main calculation)
  void makeVariables();

  // setters

  void setUse3DAngle(bool v) { use3DAngle_ = v; }

  void setSignalConeSize(double v) { signalConeSize_ = v; }

  void setIsolationConeSize(double v) { isolationConeSize_ = v; }

  void setUseVariableSignalCone(bool v) { useVariableSignalCone_ = v; }

  void setSignalConeFunction(double v) { signalConeFunction_ = v; }

  void setUseVariableIsolationCone(bool v) { useVariableIsolationCone_ = v; }

  void setIsolationConeFunction(double v) { isolationConeFunction_ = v; }

  void setSeedTrackThreshold(double v) { seedTrackThreshold_ = v; }

  void setShoulderTrackThreshold(double v) { shoulderTrackThreshold_ = v; }

  void setPi0Threshold(double v) { pi0Threshold_ = v; }

  void setDZTrackAssociation(double v) { dZTrackAssociation_ = v; }

 private:

  // tracks momentum in the signal cone
  math::XYZTLorentzVector tracksMomentum_;

  // pi0s momentum in the signal cone
  math::XYZTLorentzVector pi0sMomentum_;

  // seed track ie. highest pt track
  reco::TrackRef seedTrack_;

  // flag to use 3D angle (default is dR)
  bool use3DAngle_;

  // signal cone size
  double signalConeSize_;

  // isolation cone size
  double isolationConeSize_;

  // flag to use shrinking signal cone
  bool useVariableSignalCone_;

  // variable signal cone function (This is only valid when useVariableSignalCone_ is on)
  // This string will be used to define a function, "signalConeFunction_/x"
  double signalConeFunction_;

  // flag to use shrinking isolation cone
  bool useVariableIsolationCone_;

  // variable isolation cone function (This is only valid when useVariableIsolationCone_ is on)
  // This string will be used to define a function, "isolationConeFunction_/x"
  double isolationConeFunction_;

  // seed track pt threshold
  double seedTrackThreshold_;

  // shoulder track pt threshold
  double shoulderTrackThreshold_;

  // pi0 et threshold
  double pi0Threshold_;

  // max distance allowed for seed track and shoulder track association
  double dZTrackAssociation_;

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

};
#endif

