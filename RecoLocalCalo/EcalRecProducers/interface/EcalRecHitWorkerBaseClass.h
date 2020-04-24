#ifndef RecoLocalCalo_EcalRecAlgos_EcalRecHitWorkerBaseClass_hh
#define RecoLocalCalo_EcalRecAlgos_EcalRecHitWorkerBaseClass_hh

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
}

class EcalRecHitWorkerBaseClass {
        public:
                enum RecoveryFlags { EB_single, EE_single, EB_VFE, EE_VFE, EB_FE, EE_FE };

                EcalRecHitWorkerBaseClass(const edm::ParameterSet&, edm::ConsumesCollector& c) {};
			    EcalRecHitWorkerBaseClass(const edm::ParameterSet&) {};
                virtual ~EcalRecHitWorkerBaseClass() {};

                virtual void set(const edm::EventSetup& es) = 0;
                virtual bool run(const edm::Event& evt, const EcalUncalibratedRecHit& uncalibRH, EcalRecHitCollection & result) = 0;
		
};

#endif
