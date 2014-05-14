#ifndef RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitWorkerBaseClass_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitWorkerBaseClass_hh

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
}

class HGCalUncalibRecHitWorkerBaseClass {
        public:
                HGCalUncalibRecHitWorkerBaseClass(const edm::ParameterSet&){}
                virtual ~HGCalUncalibRecHitWorkerBaseClass(){}

                virtual void set(const edm::EventSetup& es) = 0;
                virtual bool run1(const edm::Event& evt, const HGCEEDigiCollection::const_iterator & digi, HGCeeUncalibratedRecHitCollection & result) = 0;
                virtual bool run2(const edm::Event& evt, const HGCHEDigiCollection::const_iterator & digi, HGChefUncalibratedRecHitCollection & result) = 0;
                virtual bool run3(const edm::Event& evt, const HGCHEDigiCollection::const_iterator & digi, HGChebUncalibratedRecHitCollection & result) = 0;
};

#endif
