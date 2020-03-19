
#ifndef ElectronEnergyCorrector_H
#define ElectronEnergyCorrector_H

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class EcalClusterFunctionBaseClass;

namespace egamma {

  float classBasedElectronEnergy(reco::GsfElectron const &,
                                 reco::BeamSpot const &,
                                 EcalClusterFunctionBaseClass const &crackCorrectionFunction);
  double classBasedElectronEnergyUncertainty(reco::GsfElectron const &);
  double simpleElectronEnergyUncertainty(reco::GsfElectron const &);

}  // namespace egamma

#endif
