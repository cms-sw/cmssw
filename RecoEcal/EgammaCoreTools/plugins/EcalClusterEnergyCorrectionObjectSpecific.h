
#ifndef RecoEcal_EgammaCoreTools_EcalClusterEnergyCorrection_ObjectSpecific_h
#define RecoEcal_EgammaCoreTools_EcalClusterEnergyCorrection_ObjectSpecific_h


/** \class EcalClusterEnergyCorrectionObjectSpecific
  *  Function that provides supercluster energy correction due to Bremsstrahlung loss
  *
  *  $Id: EcalClusterEnergyCorrectionObjectSpecific.h
  *  $Date:
  *  $Revision:
  *  \author Nicolas Chanon, October 2011
  */

//#include "DataFormats/EgammaCandidates/interface/Photon.h"
//#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterEnergyCorrectionObjectSpecificBaseClass.h"

class EcalClusterEnergyCorrectionObjectSpecific : public EcalClusterEnergyCorrectionObjectSpecificBaseClass {
        public:
                EcalClusterEnergyCorrectionObjectSpecific( const edm::ParameterSet &){};
                // compute the correction

		//float getValue( const reco::Photon &, const int mode) const;
		//virtual float getValue( const reco::GsfElectron &, const int mode) const;

                virtual float getValue( const reco::SuperCluster &, const int mode) const;
                virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const { return 0.;};

	        float fEta  (float energy, float eta, int algorithm) const;
		//float fBrem (float e,  float eta, int algorithm) const;
		//float fEtEta(float et, float eta, int algorithm) const;
		float fBremEta(float sigmaPhiSigmaEta, float eta, int algorithm) const;
		float fEt(float et, int algorithm) const;
		float fEnergy(float e, int algorithm) const;


		//float r9;
		//float e5x5;

};


#endif
