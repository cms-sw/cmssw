#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerBaseClass_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerBaseClass_hh

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
}

class EcalUncalibRecHitWorkerBaseClass {
        public:
                EcalUncalibRecHitWorkerBaseClass(const edm::ParameterSet&){}
                virtual ~EcalUncalibRecHitWorkerBaseClass(){}

                virtual void set(const edm::EventSetup& es) = 0;
                virtual bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result) = 0;
};

#endif
