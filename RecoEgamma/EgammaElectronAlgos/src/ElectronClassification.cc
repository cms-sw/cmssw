#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronClassification.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"


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

  if (electron.isEBEEGap() || electron.isEBEtaGap() || electron.isEERingGap())
   {
	electronClass_ = GsfElectron::GAP ;
	return ;
   }

  float pin  = electron.trackMomentumAtVtx().R() ;
  float fbrem = electron.fbrem() ;
  int nbrem = electron.numberOfBrems() ;

  // golden
  if (nbrem == 0 && (pin - scEnergy)/pin < 0.1 && fbrem < 0.5) {
	  electronClass_ = GsfElectron::GOLDEN ;
  }
  
  // big brem
  else if (nbrem == 0 && (pin - scEnergy)/pin < 0.1 && fbrem > 0.5) {
	  electronClass_ = GsfElectron::BIGBREM ;
  }
  
  // showering
  else 
          electronClass_ = GsfElectron::SHOWERING ;

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
