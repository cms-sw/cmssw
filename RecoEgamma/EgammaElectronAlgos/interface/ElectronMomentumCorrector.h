#ifndef ElectronMomentumCorrector_H
#define ElectronMomentumCorrector_H

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university
//         Ivica Puljak - FESB, Split
// 12/2005
//adapted to CMSSW by U.Berthon, dec 2006
//===================================================================

/*! \file ElectronMomentumCorrector.h
  Egamma class for Correction of electrons energy.
*/

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/Math/interface/LorentzVector.h"

class ElectronMomentumCorrector
{
 public:

  //  ElectronMomentumCorrector(){newMomentum_=CLHEP::HepLorentzVector();}
   // ElectronMomentumCorrector(){ newMomentum_= math::XYZTLorentzVector();}
   ElectronMomentumCorrector() {}
   //virtual
   ~ElectronMomentumCorrector(){}

   //virtual
   void correct(reco::GsfElectron &, TrajectoryStateOnSurface &);

  //DC: USED ??
  ////  CLHEP::HepLorentzVector getBestMomentum() const {return newMomentum_;}
  //math::XYZTLorentzVector getBestMomentum() const {return newMomentum_;}
  //float getSCEnergyError() const {return errorEnergy_;}
  //float getTrackMomentumError() const {return errorTrackMomentum_;}

 private:

  float energyError(float E, float *par) const;

  //  CLHEP::HepLorentzVector newMomentum_;
  //math::XYZTLorentzVector newMomentum_;
  //float errorEnergy_;
  //float errorTrackMomentum_;

};

#endif




