#ifndef RecoEcal_EgammaCoreTools_EcalClusterCrackCorrectionBaseClass_h
#define RecoEcal_EgammaCoreTools_EcalClusterCrackCorrectionBaseClass_h

/** \class EcalClusterCrackCorrection
  *  Function to correct cluster for cracks in the calorimeter
  *
  *  $Id: EcalClusterCrackCorrection.h
  *  $Date:
  *  $Revision:
  *  \author Federico Ferri, CEA Saclay, November 2008
  */

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalClusterCrackCorrParameters.h"

class EcalClusterCrackCorrectionBaseClass : public EcalClusterFunctionBaseClass {
        public:
                EcalClusterCrackCorrectionBaseClass();
                EcalClusterCrackCorrectionBaseClass( const edm::ParameterSet & ) {};
                virtual ~EcalClusterCrackCorrectionBaseClass();

                // get/set explicit methods for parameters
                const EcalClusterCrackCorrParameters * getParameters() const { return params_; }
                // check initialization
                void checkInit() const;
                
                // compute the correction
                virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const = 0;
                virtual float getValue( const reco::SuperCluster &, const int mode ) const = 0;
		
		virtual float getValue( const reco::CaloCluster &) const{return 0;};
                // set parameters
                virtual void init( const edm::EventSetup& es );

        protected:
                edm::ESHandle<EcalClusterCrackCorrParameters> esParams_;
                const EcalClusterCrackCorrParameters * params_;
		const edm::EventSetup * es_; //needed to access the ECAL geometry

};

#endif
