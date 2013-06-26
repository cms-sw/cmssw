#ifndef RecoEcal_EgammaCoreTools_EcalClusterEnergyUncertainty_h
#define RecoEcal_EgammaCoreTools_EcalClusterEnergyUncertainty_h

/** \class EcalClusterEnergyUncertainty
  *  Function that provides uncertainty on supercluster energy measurement
  *  Available numbers: total effective uncertainty (in GeV)
  *                     assymetric uncertainties (positive and negative)
  *
  *  $Id: EcalClusterEnergyUncertainty.h
  *  $Date:
  *  $Revision:
  *  \author Yurii Maravin, KSU, March 2009
  */

#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterEnergyUncertaintyBaseClass.h"

class EcalClusterEnergyUncertainty : public EcalClusterEnergyUncertaintyBaseClass {
        public:
                EcalClusterEnergyUncertainty( const edm::ParameterSet &){};
                // compute the correction
                virtual float getValue( const reco::SuperCluster &, const int mode ) const;
                virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const { return 0.;};
	
};

#endif
