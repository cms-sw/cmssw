#ifndef RecoEgamma_EgammaElectronAlgos_ecalClusterEnergyUncertaintyElectronSpecific_h
#define RecoEgamma_EgammaElectronAlgos_ecalClusterEnergyUncertaintyElectronSpecific_h

/** ecalClusterEnergyUncertaintyElectronSpecific
  *  Function that provides uncertainty on supercluster energy measurement
  *  Available numbers: total effective uncertainty (in GeV)
  *                     assymetric uncertainties (positive and negative)
  *
  *  $Id: ecalClusterEnergyUncertaintyElectronSpecific.h
  *  $Date:
  *  $Revision:
  *  \author Nicolas Chanon, December 2011
  */

namespace reco {
  class SuperCluster;
}

namespace egamma {

  float ecalClusterEnergyUncertaintyElectronSpecific(reco::SuperCluster const& superCluster);

}

#endif
