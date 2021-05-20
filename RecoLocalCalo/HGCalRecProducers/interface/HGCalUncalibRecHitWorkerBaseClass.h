#ifndef RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitWorkerBaseClass_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitWorkerBaseClass_hh

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}  // namespace edm

// this worker class structure is not well thought out and needs to
// change in the future.
class HGCalUncalibRecHitWorkerBaseClass {
public:
  HGCalUncalibRecHitWorkerBaseClass(const edm::ParameterSet& ps, edm::ConsumesCollector iC) {}
  virtual ~HGCalUncalibRecHitWorkerBaseClass() {}

  // do event setup things
  virtual void set(const edm::EventSetup& es) = 0;

  // run HGC-EE things
  virtual bool runHGCEE(const HGCalDigiCollection::const_iterator& digi, HGCeeUncalibratedRecHitCollection& result) = 0;

  // run HGC-FH things
  virtual bool runHGCHEsil(const HGCalDigiCollection::const_iterator& digi,
                           HGChefUncalibratedRecHitCollection& result) = 0;

  // run HGC-BH things
  virtual bool runHGCHEscint(const HGCalDigiCollection::const_iterator& digi,
                             HGChebUncalibratedRecHitCollection& result) = 0;

  // run HFNose things
  virtual bool runHGCHFNose(const HGCalDigiCollection::const_iterator& digi,
                            HGChfnoseUncalibratedRecHitCollection& result) = 0;
};

#endif
