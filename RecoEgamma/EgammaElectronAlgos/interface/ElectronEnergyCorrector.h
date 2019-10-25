
#ifndef ElectronEnergyCorrector_H
#define ElectronEnergyCorrector_H

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class EcalClusterFunctionBaseClass;

namespace electronAlgos {

  float classBasedParameterizationEnergy(reco::GsfElectron const &,
                                         reco::BeamSpot const &,
                                         EcalClusterFunctionBaseClass const &crackCorrectionFunction);
  double classBasedParameterizationUncertainty(reco::GsfElectron const &);
  double simpleParameterizationUncertainty(reco::GsfElectron const &);

}  // namespace electronAlgos

#endif
