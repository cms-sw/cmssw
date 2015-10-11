#ifndef RecoLocalCalo_HGCalRecProducers_HGCalRecHitWorkerBaseClass_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalRecHitWorkerBaseClass_hh

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
}

class HGCalRecHitWorkerBaseClass {
        public:

                HGCalRecHitWorkerBaseClass(const edm::ParameterSet&) {};
                virtual ~HGCalRecHitWorkerBaseClass() {};

                virtual void set(const edm::EventSetup& es) = 0;
                virtual bool run(const edm::Event& evt, const HGCUncalibratedRecHit& uncalibRH, HGCRecHitCollection & result) = 0;
};

#endif
