#ifndef RecoLocalCalo_HGCalRecProducers_HGCalRecHitWorkerBaseClass_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalRecHitWorkerBaseClass_hh

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}  // namespace edm

class HGCalRecHitWorkerBaseClass {
public:
  HGCalRecHitWorkerBaseClass(const edm::ParameterSet&, edm::ConsumesCollector){};
  virtual ~HGCalRecHitWorkerBaseClass(){};

  virtual void set(const edm::EventSetup& es) = 0;
  virtual void run(const edm::Event& evt,
                   const HGCUncalibratedRecHitCollection& uncalibRH,
                   HGCRecHitCollection& result) = 0;
};

#endif
