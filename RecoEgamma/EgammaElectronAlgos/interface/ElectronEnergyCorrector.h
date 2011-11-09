
#ifndef ElectronEnergyCorrector_H
#define ElectronEnergyCorrector_H

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class ElectronEnergyCorrector
 {
  public:

    ElectronEnergyCorrector() {}

    void correctEcalEnergy( reco::GsfElectron &, const reco::BeamSpot & bs /*, bool applyEcalEnergyCorrection = true*/ ) ;
    void correctEcalEnergyError( reco::GsfElectron & ) ;

  private:

    //void computeNewEnergy( const reco::GsfElectron &, bool applyEcalEnergyCorrection = true ) ;

    double fEtaBarrelBad( double scEta ) const ;
    double fEtaBarrelGood( double scEta ) const ;
    double fEtaEndcapBad( double scEta ) const ;
    double fEtaEndcapGood( double scEta ) const ;

    // new corrections (N. Chanon et al.)
    float fEta  (float energy, float eta, int algorithm) const ;
    //float fBrem (float e,  float eta, int algorithm) const ;
    //float fEtEta(float et, float eta, int algorithm) const ;
    float fBremEta(float sigmaPhiSigmaEta, float eta, int algorithm, reco::GsfElectron::Classification cl ) const ;
    float fEt(float et, int algorithm, reco::GsfElectron::Classification cl ) const ;
    float fEnergy(float e, int algorithm, reco::GsfElectron::Classification cl ) const ;

   //EcalClusterFunctionBaseClass * ff_ ;
   //float newEnergy_ ;
   //float newEnergyError_ ;

 } ;

#endif




