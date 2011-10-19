
#ifndef ElectronEnergyCorrector_H
#define ElectronEnergyCorrector_H

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class ElectronEnergyCorrector
 {
  public:

    ElectronEnergyCorrector()
     {}

    void correct( reco::GsfElectron &, const reco::BeamSpot & bs, bool applyEcalEnergyCorrection = true ) ;
    void setCorrectedEcalEnergyError( reco::GsfElectron & ) ;

  private:

    //void computeNewEnergy( const reco::GsfElectron &, bool applyEcalEnergyCorrection = true ) ;

    double fEtaBarrelBad( double scEta ) const ;
    double fEtaBarrelGood( double scEta ) const ;
    double fEtaEndcapBad( double scEta ) const ;
    double fEtaEndcapGood( double scEta ) const ;

  //  EcalClusterFunctionBaseClass * ff_ ;
  //  float newEnergy_ ;
  //  float newEnergyError_ ;

 } ;

#endif




