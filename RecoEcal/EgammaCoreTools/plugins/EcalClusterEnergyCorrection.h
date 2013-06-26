#ifndef RecoEcal_EgammaCoreTools_EcalClusterEnergyCorrection_h
#define RecoEcal_EgammaCoreTools_EcalClusterEnergyCorrection_h

/** \class EcalClusterEnergyCorrection
  *  Function that provides supercluster energy correction due to Bremsstrahlung loss
  *
  *  $Id: EcalClusterEnergyCorrection.h
  *  $Date:
  *  $Revision:
  *  \author Yurii Maravin, KSU, March 2009
  */

#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterEnergyCorrectionBaseClass.h"

class EcalClusterEnergyCorrection : public EcalClusterEnergyCorrectionBaseClass {
        public:
                EcalClusterEnergyCorrection( const edm::ParameterSet &){};
                // compute the correction
                virtual float getValue( const reco::SuperCluster &, const int mode ) const;
                virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const { return 0.;};

	        float fEta  (float e,  float eta, int algorithm) const;
		float fBrem (float e,  float eta, int algorithm) const;
		float fEtEta(float et, float eta, int algorithm) const;
};

#endif
