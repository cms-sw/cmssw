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
// new classification numbering and showering subclasses
// golden                     =>  0
// big brem                   => 10
// narrow                     => 20
// showering nbrem 0          => 30
// showering nbrem 1          => 31
// showering nbrem 2          => 32
// showering nbrem 3          => 33
// showering nbrem 4 ou plus  => 34
// cracks                     => 40
// endcap                     => barrel + 100
// CC 08/02/2006
//===================================================================

using namespace reco;

void ElectronClassification::correct(GsfElectron &electron) {
  classify(electron);
  //  electron.classifyElectron(this);
  electron.classifyElectron(electronClass_);
}

void ElectronClassification::classify(const GsfElectron &electron) {

   reco::SuperClusterRef sclRef=electron.superCluster();

  // use supercluster energy including f(Ncry) correction
  float scEnergy=sclRef->energy();
    
  // first look whether it's in crack, barrel or endcap
  std::vector<DetId> vecId=sclRef->seed()->getHitsByDetId();
  int detector =vecId[0].subdetId();  
  if (isInCrack(fabs(electron.eta()))) {
    electronClass_=40;
    return;
  } else if (detector==EcalBarrel) {
    electronClass_ = 0;
  } else if (detector==EcalEndcap) {
    electronClass_ = 100;
  } else {
    electronClass_=-1;
    edm::LogWarning("") << "ElectronClassification::init(): Undefined electron, eta = " << 
      electron.eta() << "!!!!" ;
    return;
  }

  // then decide to which class it belongs
  float p0 = 7.20583e-04;
  float p1 = 9.20913e-02;
  float p2 = 8.97269e+00;

  float pin  = electron.trackMomentumAtVtx().R(); 
  float pout = electron.trackMomentumOut().R(); 
     
  float peak = p0 + p1/(pin-p2);
  
  int nbrem = electron.numberOfClusters()-1;

  // golden
  if (nbrem == 0 && 
      (pin - scEnergy)/pin < 0.1 &&
      fabs(electron.caloPosition().phi() -
	   electron.gsfTrack()->outerMomentum().phi() - peak) < 0.15 &&
      (pin - pout)/pin < 0.2) {    
    electronClass_ += 0;
  } 
  // big brem
  else if (nbrem == 0 &&
	   (pin - pout)/pin > 0.5 &&
	   fabs(pin - scEnergy)/pin < 0.1) {
    electronClass_ += 10;
  } 
  // narrow
  else if (nbrem == 0 &&
	   fabs(pin - scEnergy)/pin < 0.1) {
    electronClass_ += 20;
  }
  // showering
  else {   
    if (nbrem == 0) electronClass_ += 30;
    if (nbrem == 1) electronClass_ += 31;
    if (nbrem == 2) electronClass_ += 32;
    if (nbrem == 3) electronClass_ += 33;
    if (nbrem >= 4) electronClass_ += 34;
  }

}

bool ElectronClassification::isInCrack(float eta) const{

  return (eta < 0.018 || 
	  (eta>0.423 && eta<0.461) ||
	  (eta>0.770 && eta<0.806) ||
	  (eta>1.127 && eta<1.163) ||
	  (eta>1.460 && eta<1.558));

}
