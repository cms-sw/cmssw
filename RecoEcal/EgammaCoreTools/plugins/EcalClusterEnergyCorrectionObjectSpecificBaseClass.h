#ifndef RecoEcal_EgammaCoreTools_EcalClusterEnergyCorrectionObjectSpecificBaseClass_h
#define RecoEcal_EgammaCoreTools_EcalClusterEnergyCorrectionObjectSpecificBaseClass_h

/** \class EcalClusterEnergyCorrectionBaseClass
  *  Function to correct cluster for the so called local containment
  *
  *  $Id: EcalClusterEnergyCorrectionBaseClass.h
  *  $Date:
  *  $Revision:
  *  \author Nicolas Chanon, October 2011
  */

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"

//#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionObjectSpecificParameters.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
//#include "DataFormats/EgammaCandidates/interface/Photon.h"
//#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

namespace edm {
        class EventSetup;
        class ParameterSet;
}

class EcalClusterEnergyCorrectionObjectSpecificBaseClass : public EcalClusterFunctionBaseClass {
        public:
                EcalClusterEnergyCorrectionObjectSpecificBaseClass();
                EcalClusterEnergyCorrectionObjectSpecificBaseClass( const edm::ParameterSet & ) {};
                virtual ~EcalClusterEnergyCorrectionObjectSpecificBaseClass();

                // get/set explicit methods for parameters
                const EcalClusterEnergyCorrectionObjectSpecificParameters * getParameters() const { return params_; }
                // check initialization
                void checkInit() const;
                
                // compute the correction
                virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const = 0;
                virtual float getValue( const reco::SuperCluster &, const int mode ) const = 0;
		
		//virtual float getValue( const reco::Photon &, const int mode) const = 0;
		//virtual float getValue( const reco::GsfElectron &, const int mode) const;


                // set parameters
                virtual void init( const edm::EventSetup& es );

        protected:
                edm::ESHandle<EcalClusterEnergyCorrectionObjectSpecificParameters> esParams_;
                const EcalClusterEnergyCorrectionObjectSpecificParameters * params_;
};

#endif
