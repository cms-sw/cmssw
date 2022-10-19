#ifndef RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitWorkerBaseClass_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitWorkerBaseClass_hh

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

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

  // run HGC-EE things
  virtual bool runHGCEE(const edm::ESHandle<HGCalGeometry>& geom,
                        const HGCalDigiCollection& digis,
                        HGCeeUncalibratedRecHitCollection& result) = 0;

  // run HGC-FH things
  virtual bool runHGCHEsil(const edm::ESHandle<HGCalGeometry>& geom,
                           const HGCalDigiCollection& digis,
                           HGChefUncalibratedRecHitCollection& result) = 0;

  // run HGC-BH things
  virtual bool runHGCHEscint(const edm::ESHandle<HGCalGeometry>& geom,
                             const HGCalDigiCollection& digis,
                             HGChebUncalibratedRecHitCollection& result) = 0;

  // run HFNose things
  virtual bool runHGCHFNose(const edm::ESHandle<HGCalGeometry>& geom,
                            const HGCalDigiCollection& digis,
                            HGChfnoseUncalibratedRecHitCollection& result) = 0;
};

#endif
