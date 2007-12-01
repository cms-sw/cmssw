#ifndef PixelTrackProducerWithZPos_H
#define PixelTrackProducerWithZPos_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"

class PixelFitter;
class PixelTrackCleaner;
class TrackHitsFilter;
class OrderedHitsGenerator;
class TrackingRegionProducer;

//class GlobalTrackingRegion;
//class SiPixelRecHitCollection;

namespace edm { class Event; class EventSetup; }

class PixelTrackProducerWithZPos :  public edm::EDProducer
{
  public:
    explicit PixelTrackProducerWithZPos(const edm::ParameterSet& conf);
    ~PixelTrackProducerWithZPos();
    virtual void produce(edm::Event& ev, const edm::EventSetup& es);
 
  private:
//    SiPixelRecHitCollection getHits(const edm::Event& ev);
    void beginJob(const edm::EventSetup& es);
    void store(edm::Event& ev,
               const pixeltrackfitting::TracksWithRecHits & selectedTracks);

    edm::ParameterSet theConfig;

    const PixelFitter       * theFitter;
    const TrackHitsFilter   * theFilter;
          PixelTrackCleaner * theCleaner;  
          OrderedHitsGenerator * theGenerator;
          TrackingRegionProducer* theRegionProducer;
};
#endif
