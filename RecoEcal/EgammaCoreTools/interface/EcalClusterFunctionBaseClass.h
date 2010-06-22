#ifndef RecoEcal_EgammaCoreTools_EcalClusterFunctionBaseClass_hh
#define RecoEcal_EgammaCoreTools_EcalClusterFunctionBaseClass_hh

/** \class EcalClusterFunction
  *  Base class for all functions needed to e.g.
  *  correct cracks/gap, estimate cluster energy error etc.
  *
  *  $Id: EcalClusterFunction.h
  *  $Date: 
  *  $Revision: 
  *  \author Federico Ferri, CEA Saclay, November 2008
  */

//#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
}


class EcalClusterFunctionBaseClass {
        public:
                virtual ~EcalClusterFunctionBaseClass() {};
                virtual void  init( const edm::EventSetup& es ) = 0;
                virtual float getValue( const reco::BasicCluster &, const EcalRecHitCollection & ) const = 0;
                virtual float getValue( const reco::SuperCluster &, const int mode ) const = 0;
	        //this one is needed for EcalClusterCrackCorrection:
	        virtual float getValue( const reco::CaloCluster &) const {return 0;};

};

#endif
