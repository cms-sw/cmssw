#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronClassification.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university
// 12/2005
// See GsfElectron::Classification
//===================================================================

using namespace reco;

void ElectronClassification::correct(GsfElectron &electron) {
  classify(electron);
  //  electron.classifyElectron(this);
  electron.classifyElectron(electronClass_);
}

void ElectronClassification::classify(const GsfElectron &electron)
 {
  electronClass_ = GsfElectron::UNKNOWN ;


   reco::SuperClusterRef sclRef=electron.superCluster();

  // use supercluster energy including f(Ncry) correction
  float scEnergy=sclRef->energy();

  // first look whether it's in crack, barrel or endcap
  if ((!electron.isEB())&&(!electron.isEE()))
   {
    edm::LogWarning("") << "ElectronClassification::init(): Undefined electron, eta = " <<
      electron.eta() << "!!!!" ;
    return ;
   }

  if (electron.isGap())
   {
	electronClass_ = GsfElectron::GAP ;
	return ;
   }

//  // cracks
//  if (electron.isEBEEGap()) {
//    electronClass_+=40;
//    return;
//  } else if (electron.isEBEtaGap() || electron.isEERingGap()) {
//    electronClass_+=41;
//    return;
//  } else if (electron.isEBPhiGap() || electron.isEEDeeGap()) {
//    electronClass_+=42;
//    return;
//  }

  // then decide for the others to which class it belongs
  float p0 = 7.20583e-04;
  float p1 = 9.20913e-02;
  float p2 = 8.97269e+00;

  float pin  = electron.trackMomentumAtVtx().R() ;
  float fbrem = electron.fbrem() ;
  float peak = p0 + p1/(pin-p2) ;
  int nbrem = electron.numberOfBrems() ;

  // golden
  if (nbrem == 0 &&
      (pin - scEnergy)/pin < 0.1 &&
      fabs(electron.caloPosition().phi() -
	   electron.gsfTrack()->outerMomentum().phi() - peak) < 0.15 &&
      fbrem < 0.2) {
	  electronClass_ = GsfElectron::GOLDEN ;
  }
  // big brem
  else if (nbrem == 0 &&
	   fbrem > 0.5 &&
	   fabs(pin - scEnergy)/pin < 0.1) {
	  electronClass_ = GsfElectron::BIGBREM ;
  }
  // narrow
  else if (nbrem == 0 &&
	   fabs(pin - scEnergy)/pin < 0.1) {
    electronClass_ = GsfElectron::NARROW ;
  }
  // showering
  else {
	    electronClass_ = GsfElectron::SHOWERING ;
	  }
//    if (nbrem == 0) electronClass_ += 30;
//    if (nbrem == 1) electronClass_ += 31;
//    if (nbrem == 2) electronClass_ += 32;
//    if (nbrem == 3) electronClass_ += 33;
//    if (nbrem >= 4) electronClass_ += 34;
//  }

}

/*
bool ElectronClassification::isInCrack(float eta) const{

  return (eta>1.460 && eta<1.558);

}

bool ElectronClassification::isInEtaGaps(float eta) const{

  return (eta < 0.018 ||
	  (eta>0.423 && eta<0.461) ||
	  (eta>0.770 && eta<0.806) ||
	  (eta>1.127 && eta<1.163));

}

bool ElectronClassification::isInPhiGaps(float phi) const{

  return false;

}
*/
