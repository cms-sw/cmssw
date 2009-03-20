
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronEnergyCorrector.h"

//#include "ElectronPhoton/ElectronReco/interface/PRecElectron.h"
//#include "ElectronPhoton/EgammaPreshower/interface/EgammaEndcapCluster.h"
//#include <CLHEP/Vector/LorentzVector.h>

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university
//         Ivica Puljak - FESB, Split
// 12/2005
// updated f(eta) corrections from Ivica
// CC 02/2006
// adapted for crack subdivision
// 09/2008
// ported to CMSSW by U. Berthon, dec 2006
//
//===================================================================

float energyError(float E, float *par)
 { return sqrt( pow(par[0]/sqrt(E),2) + pow(par[1]/E,2) + pow(par[2],2) ) ; }

void ElectronEnergyCorrector::correct
 ( reco::GsfElectron & electron, bool applyEtaCorrection )
 {
  if (electron.isEcalEnergyCorrected())
   {
	edm::LogWarning("ElectronEnergyCorrector::correct")<<"already done" ;
	return ;
   }

  computeNewEnergy(electron,applyEtaCorrection) ;
  electron.correctEcalEnergy(newEnergy_,newEnergyError_) ;
 }

void ElectronEnergyCorrector::computeNewEnergy
 ( const reco::GsfElectron & electron, bool applyEtaCorrection )
 {
  double scEnergy = electron.superCluster()->energy() ;
  int elClass = electron.classification() ;
  newEnergy_ = scEnergy ;
  newEnergyError_ = electron.ecalEnergyError() ;

  //===================
  // irrelevant classification
  //===================

  if ( (elClass <= reco::GsfElectron::UNKNOWN) ||
	   (elClass>reco::GsfElectron::GAP) )
   {
	edm::LogWarning("ElectronMomentumCorrector::correct")<<"unexpected classification" ;
	return ;
   }

  //===================
  // If not gap, f(eta) correction ;
  //===================

  if ( applyEtaCorrection && (!electron.isGap()) )
   {
    double scEta = electron.caloPosition().eta() ;
    if (electron.isEB()) // barrel
     {
	  if ( (elClass==reco::GsfElectron::GOLDEN) ||
	       (elClass==reco::GsfElectron::BIGBREM) ||
	       (elClass==reco::GsfElectron::NARROW) )
	   { newEnergy_ = scEnergy/fEtaBarrelGood(scEta) ; }
	  else if (elClass==reco::GsfElectron::SHOWERING)
	   { newEnergy_ = scEnergy/fEtaBarrelBad(scEta) ; }
     }
    else if (electron.isEE()) // endcap
     {
      double ePreshower = electron.superCluster()->preshowerEnergy() ;
      if ( (elClass==reco::GsfElectron::GOLDEN) ||
	       (elClass==reco::GsfElectron::BIGBREM) ||
	       (elClass==reco::GsfElectron::NARROW) )
       { newEnergy_ = (scEnergy-ePreshower)/fEtaEndcapGood(scEta)+ePreshower ; }
	  else if (elClass==reco::GsfElectron::SHOWERING)
	   { newEnergy_ = (scEnergy-ePreshower)/fEtaEndcapBad(scEta)+ePreshower ; }
     }
    else
     { edm::LogWarning("ElectronEnergyCorrector::computeNewEnergy")<<"nor barrel neither endcap electron !" ; }
   }

  //===================
  // energy error
  //=====================

  // steph third sigma
  float parEB[5][3] =
   {
     { 2.46e-02,  1.97e-01, 5.23e-03},          // golden
     { 9.99e-07,  2.80e-01, 5.69e-03},          // big brem
     { 9.37e-07,  2.32e-01, 5.82e-03},          // narrow
     { 7.30e-02,  1.95e-01, 1.30e-02},          // showering
     { 9.25e-06,  2.84e-01, 8.77e-03}           // nominal --> gap
   } ;

  float parEE[5][3] =
   {
     { 1.25841e-01, 7.01145e-01, 2.81884e-11},  // golden
     { 1.25841e-01, 7.01145e-01, 2.81884e-11},  // big brem = golden
     { 1.25841e-01, 7.01145e-01, 2.81884e-11},  // narrow = golden
     { 1.63634e-01, 1.11307e+00, 3.64770e-03},  // showering
     {         .02,         .15,        .005}   // nominal --> gap
   } ;

  if (electron.isEB())
   { newEnergyError_ =  scEnergy * energyError(scEnergy,parEB[elClass]) ; }
  else if (electron.isEE())
   { newEnergyError_ =  scEnergy * energyError(scEnergy,parEE[elClass]) ; }
  else
   { edm::LogWarning("ElectronEnergyCorrector::computeNewEnergy")<<"nor barrel neither endcap electron !" ; }

 }


double ElectronEnergyCorrector::fEtaBarrelGood( double scEta ) const
 {
  // f(eta) for the first 3 classes (0, 10 and 20) (estimated from 1Mevt single e sample)
  // Ivica's new corrections 01/06
  float p0 =  1.00149e+00 ;
  float p1 = -2.06622e-03 ;
  float p2 = -1.08793e-02 ;
  float p3 =  1.54392e-02 ;
  float p4 = -1.02056e-02 ;
  double x  = (double) fabs(scEta) ;
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x ;
 }

double ElectronEnergyCorrector::fEtaBarrelBad(double scEta) const
 {
  // f(eta) for the class = 30 (estimated from 1Mevt single e sample)
  // Ivica's new corrections 01/06
  float p0 =  9.99063e-01;
  float p1 = -2.63341e-02;
  float p2 =  5.16054e-02;
  float p3 = -4.95976e-02;
  float p4 =  3.62304e-03;
  double x  = (double) fabs(scEta) ;
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x ;
 }

double ElectronEnergyCorrector::fEtaEndcapGood( double scEta ) const
 {
  // f(eta) for the first 3 classes (100, 110 and 120)
  // Ivica's new corrections 01/06
  float p0 = -8.51093e-01 ;
  float p1 =  3.54266e+00 ;
  float p2 = -2.59288e+00 ;
  float p3 = 8.58945e-01 ;
  float p4 = -1.07844e-01 ;
  double x  = (double) fabs(scEta) ;
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x ;
 }

double ElectronEnergyCorrector::fEtaEndcapBad( double scEta ) const
 {
  // f(eta) for the class = 130-134
  // Ivica's new corrections 01/06
  float p0 =        -4.25221e+00 ;
  float p1 =         1.01936e+01 ;
  float p2 =        -7.48247e+00 ;
  float p3 =         2.45520e+00 ;
  float p4 =        -3.02872e-01 ;
  double x  = (double) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x ;
 }




