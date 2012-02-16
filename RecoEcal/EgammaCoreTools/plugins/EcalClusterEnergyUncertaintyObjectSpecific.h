#ifndef RecoEcal_EgammaCoreTools_EcalClusterEnergyUncertaintyObjectSpecific_h
#define RecoEcal_EgammaCoreTools_EcalClusterEnergyUncertaintyObjectSpecific_h

/** \class EcalClusterEnergyUncertainty
  *  Function that provides uncertainty on supercluster energy measurement
  *  Available numbers: total effective uncertainty (in GeV)
  *                     assymetric uncertainties (positive and negative)
  *
  *  $Id: EcalClusterEnergyUncertainty.h
  *  $Date:
  *  $Revision:
  *  \author Nicolas Chanon, December 2011
  */

#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterEnergyUncertaintyObjectSpecificBaseClass.h"

class EcalClusterEnergyUncertaintyObjectSpecific : public EcalClusterEnergyUncertaintyObjectSpecificBaseClass {
        public:
                EcalClusterEnergyUncertaintyObjectSpecific( const edm::ParameterSet &){};
                // compute the correction
                virtual float getValue( const reco::SuperCluster &, const int mode ) const;
                virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const { return 0.;};
	
};

#endif
