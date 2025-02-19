#ifndef RecoEcal_EgammaCoreTools_EcalClusterEnergyUncertaintyBaseClass_h
#define RecoEcal_EgammaCoreTools_EcalClusterEnergyUncertaintyBaseClass_h

/** \class EcalClusterEnergyUncertaintyBaseClass
  *  Function to correct cluster for the so called local containment
  *
  *  $Id: EcalClusterEnergyUncertaintyBaseClass.h
  *  $Date:
  *  $Revision:
  *  \author Yurii Maravin, KSU, March 20, 2009
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

class EcalClusterEnergyUncertaintyBaseClass : public EcalClusterFunctionBaseClass {
        public:
                EcalClusterEnergyUncertaintyBaseClass();
                EcalClusterEnergyUncertaintyBaseClass( const edm::ParameterSet & ) {};
                virtual ~EcalClusterEnergyUncertaintyBaseClass();

                // get/set explicit methods for parameters
                const EcalClusterEnergyUncertaintyParameters * getParameters() const { return params_; }
                // check initialization
                void checkInit() const;
                
                // compute the correction
                virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const = 0;
                virtual float getValue( const reco::SuperCluster &, const int mode ) const = 0;


                // set parameters
                virtual void init( const edm::EventSetup& es );

        protected:
                edm::ESHandle<EcalClusterEnergyUncertaintyParameters> esParams_;
                const EcalClusterEnergyUncertaintyParameters * params_;
};

#endif
