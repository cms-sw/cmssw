
#ifndef ElectronEnergyCorrector_H
#define ElectronEnergyCorrector_H

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"

class ElectronEnergyCorrector
{
 public:

  ElectronEnergyCorrector( EcalClusterFunctionBaseClass * ff =0 )
   : ff_(ff) {}

  void correct(reco::GsfElectron &, bool applyEtaCorrection = true);

 private:

  //void computeNewEnergy( const reco::GsfElectron &, bool applyEtaCorrection = true ) ;

  double fEtaBarrelBad( double scEta ) const ;
  double fEtaBarrelGood( double scEta ) const ;
  double fEtaEndcapBad( double scEta ) const ;
  double fEtaEndcapGood( double scEta ) const ;

  EcalClusterFunctionBaseClass * ff_ ;
//  float newEnergy_ ;
//  float newEnergyError_ ;

};

#endif




