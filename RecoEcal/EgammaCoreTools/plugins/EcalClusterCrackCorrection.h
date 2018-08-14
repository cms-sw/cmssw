#ifndef RecoEcal_EgammaCoreTools_EcalClusterCrackCorrection_h
#define RecoEcal_EgammaCoreTools_EcalClusterCrackCorrection_h

/** \class EcalClusterCrackCorrection
  *  Function to correct cluster for cracks in the calorimeter
  *
  *  $Id: EcalClusterCrackCorrection.h
  *  $Date:
  *  $Revision:
  *  \author Federico Ferri, CEA Saclay, November 2008
  */

#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterCrackCorrectionBaseClass.h"

class EcalClusterCrackCorrection : public EcalClusterCrackCorrectionBaseClass {
        public:
                EcalClusterCrackCorrection( const edm::ParameterSet &) {};
                // compute the correction
                float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const override;
                float getValue( const reco::SuperCluster &, const int mode ) const override;

		float getValue( const reco::CaloCluster &) const override;

		
};
#endif
