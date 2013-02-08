#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronMomentumCorrector.h"


#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

/****************************************************************************
 *
 * Class based E-p combination for the final electron momentum. It relies on
 * the electron classification and on the dtermination of track momentum and ecal
 * supercluster energy errors. The track momentum error is taken from the gsf fit.
 * The ecal supercluster energy error is taken from a class dependant parametrisation
 * of the energy resolution.
 *
 *
 * \author Federico Ferri - INFN Milano, Bicocca university
 * \author Ivica Puljak - FESB, Split
 * \author Stephanie Baffioni - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
 *
 * \version $Id: ElectronMomentumCorrector.cc,v 1.20 2011/12/09 18:29:12 chamont Exp $
 *
 ****************************************************************************/


void ElectronMomentumCorrector::correct( reco::GsfElectron & electron, TrajectoryStateOnSurface & vtxTsos )
 {
  if (electron.p4Error(reco::GsfElectron::P4_COMBINATION)!= 999.)
   {
    edm::LogWarning("ElectronMomentumCorrector::correct")<<"already done" ;
    return ;
   }

  math::XYZTLorentzVector newMomentum = electron.p4() ; // default
  int elClass = electron.classification() ;

  // irrelevant classification
  if ( (elClass <= reco::GsfElectron::UNKNOWN) ||
	   (elClass>reco::GsfElectron::GAP) )
   {
    edm::LogWarning("ElectronMomentumCorrector::correct")<<"unexpected classification" ;
    return ;
   }

  float scEnergy = electron.correctedEcalEnergy() ;
  float errorEnergy_ = electron.correctedEcalEnergyError() ;

  float trackMomentum  = electron.trackMomentumAtVtx().R() ;
  float errorTrackMomentum = 999. ;

  // retreive momentum error
  MultiGaussianState1D qpState(MultiGaussianStateTransform::multiState1D(vtxTsos,0));
  GaussianSumUtilities1D qpUtils(qpState);
  errorTrackMomentum = trackMomentum*trackMomentum*sqrt(qpUtils.mode().variance());

  float finalMomentum = electron.p4().t(); // initial
  float finalMomentumError = 999.;


  // first check for large errors

  if (errorTrackMomentum/trackMomentum > 0.5 && errorEnergy_/scEnergy <= 0.5) {
    finalMomentum = scEnergy;    finalMomentumError = errorEnergy_;
   }
  else if (errorTrackMomentum/trackMomentum <= 0.5 && errorEnergy_/scEnergy > 0.5){
    finalMomentum = trackMomentum;  finalMomentumError = errorTrackMomentum;
   }
  else if (errorTrackMomentum/trackMomentum > 0.5 && errorEnergy_/scEnergy > 0.5){
    if (errorTrackMomentum/trackMomentum < errorEnergy_/scEnergy) {
      finalMomentum = trackMomentum; finalMomentumError = errorTrackMomentum;
     }
    else{
      finalMomentum = scEnergy; finalMomentumError = errorEnergy_;
     }
  }

  // then apply the combination algorithm
  else {

     // calculate E/p and corresponding error
    float eOverP = scEnergy / trackMomentum;
    float errorEOverP = sqrt(
			     (errorEnergy_/trackMomentum)*(errorEnergy_/trackMomentum) +
			     (scEnergy*errorTrackMomentum/trackMomentum/trackMomentum)*
			     (scEnergy*errorTrackMomentum/trackMomentum/trackMomentum));

//    if ( eOverP  > 1 + 2.5*errorEOverP )
//     {
//      finalMomentum = scEnergy; finalMomentumError = errorEnergy_;
//      if ((elClass==reco::GsfElectron::GOLDEN) && electron.isEB() && (eOverP<1.15))
//        {
//          if (scEnergy<15) {finalMomentum = trackMomentum ; finalMomentumError = errorTrackMomentum;}
//        }
//     }
//    else if ( eOverP < 1 - 2.5*errorEOverP )
//     {
//      finalMomentum = scEnergy; finalMomentumError = errorEnergy_;
//      if (elClass==reco::GsfElectron::SHOWERING)
//       {
//	      if (electron.isEB())
//	       {
//		      if (scEnergy<18)
//		       { finalMomentum = trackMomentum; finalMomentumError = errorTrackMomentum; }
//	       }
//	      else if (electron.isEE())
//	       {
//          if (scEnergy<13)
//           {finalMomentum = trackMomentum; finalMomentumError = errorTrackMomentum;}
//	       }
//	      else
//	       { edm::LogWarning("ElectronMomentumCorrector::correct")<<"nor barrel neither endcap electron ?!" ; }
//	     }
//	    else if (electron.isGap())
//	     {
//	      if (scEnergy<60)
//	       { finalMomentum = trackMomentum; finalMomentumError = errorTrackMomentum ; }
//	     }
//      }

    bool eleIsNotInCombination = false ;
    if ( (eOverP  > 1 + 2.5*errorEOverP) || (eOverP  < 1 - 2.5*errorEOverP) || (eOverP < 0.8) || (eOverP > 1.3) )
     { eleIsNotInCombination = true ; }
    if (eleIsNotInCombination)
     {
      if (eOverP > 1)
       { finalMomentum = scEnergy ; finalMomentumError = errorEnergy_ ; }
      else
       {
        if (elClass == reco::GsfElectron::GOLDEN)
         { finalMomentum = scEnergy; finalMomentumError = errorEnergy_; }
        if (elClass == reco::GsfElectron::BIGBREM)
         {
          if (scEnergy<36)
           { finalMomentum = trackMomentum ; finalMomentumError = errorTrackMomentum ; }
          else
           { finalMomentum = scEnergy ; finalMomentumError = errorEnergy_ ; }
         }
        if (elClass == reco::GsfElectron::BADTRACK)
         { finalMomentum = scEnergy; finalMomentumError = errorEnergy_ ; }
        if (elClass == reco::GsfElectron::SHOWERING)
         {
          if (scEnergy<30)
           { finalMomentum = trackMomentum ; finalMomentumError = errorTrackMomentum; }
          else
           { finalMomentum = scEnergy; finalMomentumError = errorEnergy_;}
         }
        if (elClass == reco::GsfElectron::GAP)
         {
          if (scEnergy<60)
           { finalMomentum = trackMomentum ; finalMomentumError = errorTrackMomentum ; }
          else
           { finalMomentum = scEnergy; finalMomentumError = errorEnergy_ ; }
         }
       }
     }

    else
     {
      // combination
      finalMomentum = (scEnergy/errorEnergy_/errorEnergy_ + trackMomentum/errorTrackMomentum/errorTrackMomentum) /
        (1/errorEnergy_/errorEnergy_ + 1/errorTrackMomentum/errorTrackMomentum);
      float finalMomentumVariance = 1 / (1/errorEnergy_/errorEnergy_ + 1/errorTrackMomentum/errorTrackMomentum);
      finalMomentumError = sqrt(finalMomentumVariance);
     }

  }

  math::XYZTLorentzVector oldMomentum = electron.p4() ;
  newMomentum = math::XYZTLorentzVector
   ( oldMomentum.x()*finalMomentum/oldMomentum.t(),
     oldMomentum.y()*finalMomentum/oldMomentum.t(),
     oldMomentum.z()*finalMomentum/oldMomentum.t(),
     finalMomentum ) ;


  // final set
  electron.correctMomentum(newMomentum,errorTrackMomentum,finalMomentumError);

 }

