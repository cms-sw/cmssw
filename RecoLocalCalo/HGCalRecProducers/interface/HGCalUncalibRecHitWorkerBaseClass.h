#ifndef RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitWorkerBaseClass_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitWorkerBaseClass_hh

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
}


// this worker class structure is not well thought out and needs to 
// change in the future.
class HGCalUncalibRecHitWorkerBaseClass {
        public:
                HGCalUncalibRecHitWorkerBaseClass(const edm::ParameterSet&){}
                virtual ~HGCalUncalibRecHitWorkerBaseClass(){}

                // do event setup things
                virtual void set(const edm::EventSetup& es) = 0;

                // run HGC-EE things
                virtual bool run1(const edm::Event& evt, const HGCEEDigiCollection::const_iterator & digi, HGCeeUncalibratedRecHitCollection & result) = 0;

                // run HGC-FH things
                virtual bool run2(const edm::Event& evt, const HGCHEDigiCollection::const_iterator & digi, HGChefUncalibratedRecHitCollection & result) = 0;

                // run HGC-BH things
                virtual bool run3(const edm::Event& evt, const HGCBHDigiCollection::const_iterator & digi, HGChebUncalibratedRecHitCollection & result) = 0;
};

#endif
