#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronMomentumCorrector.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university
// 12/2005
// Stephanie's parametrisation
// 02/2006
// adapted for crack subdivision
// 09/2008
//adapted to CMSSW by U.Berthon,LLR Paliaseau,  dec 2006
//===================================================================


/** The electron classification.
   barrel  :   0: golden,  10: bigbrem,  20: narrow, 30-34: showering,
             (30: showering nbrem=0, 31: showering nbrem=1, 32: showering nbrem=2 ,33: showering nbrem=3, 34: showering nbrem>=4)
              40: crack, 41: eta gaps, 42: phi gaps
   endcaps : 100: golden, 110: bigbrem, 120: narrow, 130-134: showering
            (130: showering nbrem=0, 131: showering nbrem=1, 132: showering nbrem=2 ,133: showering nbrem=3, 134: showering nbrem>=4)
             140: crack
*/
void ElectronMomentumCorrector::correct(reco::GsfElectron &electron, TrajectoryStateOnSurface & vtxTsos) {

  if (electron.isMomentumCorrected())
   {
    edm::LogWarning("ElectronMomentumCorrector::correct")<<"already done" ;
	return ;
   }

  newMomentum_ = electron.p4() ; // default
  int elClass = electron.classification() ;

  // irrelevant classification
  if ( (elClass <= reco::GsfElectron::UNKNOWN) ||
	   (elClass>reco::GsfElectron::GAP) )
   {
	edm::LogWarning("ElectronMomentumCorrector::correct")<<"unexpected classification" ;
	return ;
   }

//  // steph third sigma
//  float parEB[5][3] = {
//       { 2.46e-02,  1.97e-01, 5.23e-03},          // golden
//       { 9.99e-07,  2.80e-01, 5.69e-03},          // big brem
//       { 9.37e-07,  2.32e-01, 5.82e-03},          // narrow
//       { 7.30e-02,  1.95e-01, 1.30e-02},          // showering
//       { 9.25e-06,  2.84e-01, 8.77e-03}           // nominal --> crack
//  };
//
//  float parEE[5][3] = {
//       { 1.25841e-01, 7.01145e-01, 2.81884e-11},  // golden
//       { 1.25841e-01, 7.01145e-01, 2.81884e-11},  // big brem = golden
//       { 1.25841e-01, 7.01145e-01, 2.81884e-11},  // narrow = golden
//       { 1.63634e-01, 1.11307e+00, 3.64770e-03},  // showering
//       {         .02,         .15,        .005}   // nominal --> gap
//  };
//
//  // first calculate error on energy
//  errorEnergy_ = 999. ;
//  float scEnergy = electron.ecalEnergy() ;
//  if (electron.isEB()) { // barrel
//    errorEnergy_ =  scEnergy * energyError(scEnergy,parEB[elClass]);
//  }
//  else if (electron.isEE()) { //endcap
//    errorEnergy_ =  scEnergy * energyError(scEnergy,parEE[elClass]);
//  }
//  else
//   { edm::LogWarning("ElectronMomentumCorrector::correct")<<"nor barrel neither endcap electron ?!" ; }
//
  float scEnergy = electron.ecalEnergy() ;
  errorEnergy_ = electron.ecalEnergyError() ;

  // then retrieve error on track momentum
  //  float trackMomentum  =  electron.getGsfTrack()->impactPointModeMomentum().mag();
  errorTrackMomentum_ = 999. ;
  float trackMomentum  = electron.trackMomentumAtVtx().R() ;
  // momentum error rescaling
  //  std::vector<TrajectoryStateOnSurface> vtx_loc_comp = electron.getGsfTrack()->impactPointState().components();
  MultiGaussianState1D qpState(MultiGaussianStateTransform::multiState1D(vtxTsos,0));
  GaussianSumUtilities1D qpUtils(qpState);
  errorTrackMomentum_ = trackMomentum*trackMomentum*sqrt(qpUtils.mode().variance());

  // calculate E/p and corresponding error
  float eOverP = scEnergy / trackMomentum;
  float errorEOverP = sqrt(
		   (errorEnergy_/trackMomentum)*(errorEnergy_/trackMomentum) +
		   (scEnergy*errorTrackMomentum_/trackMomentum/trackMomentum)*
		   (scEnergy*errorTrackMomentum_/trackMomentum/trackMomentum));

  // combination
  float finalMomentum = (scEnergy/errorEnergy_/errorEnergy_ + trackMomentum/errorTrackMomentum_/errorTrackMomentum_) /
                       (1/errorEnergy_/errorEnergy_ + 1/errorTrackMomentum_/errorTrackMomentum_);
  if ( eOverP  > 1 + 2.5*errorEOverP )
   {
    finalMomentum = scEnergy ;
    if ((elClass==reco::GsfElectron::GOLDEN) && (eOverP<1.15))
     {
	  if (scEnergy<15) finalMomentum = trackMomentum ;
     }
   }
  else if ( eOverP < 1 - 2.5*errorEOverP )
   {
    finalMomentum = scEnergy ;
    if (elClass==reco::GsfElectron::SHOWERING)
     {
      if (electron.isEB())
       {
	    if(scEnergy<18) finalMomentum = trackMomentum;
       }
      else if (electron.isEE())
       {
	    if(scEnergy<13) finalMomentum = trackMomentum;
       }
      else
       { edm::LogWarning("ElectronMomentumCorrector::correct")<<"nor barrel neither endcap electron ?!" ; }
     }
    else if (electron.isGap())
     {
	  if(scEnergy<60) finalMomentum = trackMomentum;
     }
   }
  float finalMomentumVariance = 1 / (1/errorEnergy_/errorEnergy_ + 1/errorTrackMomentum_/errorTrackMomentum_);
  float finalMomentumError = sqrt(finalMomentumVariance);

  //  HepLorentzVector oldMomentum = electron.fourMomentum();
  //  newMomentum_ = HepLorentzVector(
  math::XYZTLorentzVector oldMomentum = electron.p4() ;
  newMomentum_ = math::XYZTLorentzVector
   ( oldMomentum.x()*finalMomentum/oldMomentum.t(),
     oldMomentum.y()*finalMomentum/oldMomentum.t(),
	 oldMomentum.z()*finalMomentum/oldMomentum.t(),
     finalMomentum ) ;

  // final set
  electron.correctMomentum(newMomentum_,errorTrackMomentum_,finalMomentumError);
 }

float ElectronMomentumCorrector::energyError(float E, float *par) const{
  return sqrt( pow(par[0]/sqrt(E),2) + pow(par[1]/E,2) + pow(par[2],2) );
}
