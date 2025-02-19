#ifndef RecoEcal_EgammaCoreTools_EcalClusterEnergyUncertaintyObjectSpecificBaseClass_h
#define RecoEcal_EgammaCoreTools_EcalClusterEnergyUncertaintyObjectSpecificBaseClass_h

/** \class EcalClusterEnergyUncertaintyObjectSpecificBaseClass
  *  Function to correct cluster for the so called local containment
  *
  *  $Id: EcalClusterEnergyUncertaintyBaseClass.h
  *  $Date:
  *  $Revision:
  *  \author Nicolas Chanon, December 2011
  */

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"

//#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalClusterEnergyUncertaintyParameters.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"

namespace edm {
        class EventSetup;
        class ParameterSet;
}

class EcalClusterEnergyUncertaintyObjectSpecificBaseClass : public EcalClusterFunctionBaseClass {
        public:
                EcalClusterEnergyUncertaintyObjectSpecificBaseClass();
                EcalClusterEnergyUncertaintyObjectSpecificBaseClass( const edm::ParameterSet & ) {};
                virtual ~EcalClusterEnergyUncertaintyObjectSpecificBaseClass();

                // get/set explicit methods for parameters
                //const EcalClusterEnergyUncertaintyParameters * getParameters() const { return params_; }
                // check initialization
                void checkInit() const;
                
                // compute the correction
                virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const = 0;
                virtual float getValue( const reco::SuperCluster &, const int mode ) const = 0;


                // set parameters
                virtual void init( const edm::EventSetup& es );

        protected:
                //edm::ESHandle<EcalClusterEnergyUncertaintyObjectSpecificParameters> esParams_;
                //const EcalClusterEnergyUncertaintyObjectSpecificParameters * params_;
};

#endif
