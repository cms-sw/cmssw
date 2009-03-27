#ifndef PixelTrackProducer_H
#define PixelTrackProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"

class PixelFitter;
class PixelTrackCleaner;
class PixelTrackFilter;
class OrderedHitsGenerator;
class TrackingRegionProducer;

namespace edm { class Event; class EventSetup; }

class PixelTrackProducer :  public edm::EDProducer {

public:
  explicit PixelTrackProducer(const edm::ParameterSet& conf);

  ~PixelTrackProducer();

  virtual void beginRun(edm::Run &run, const edm::EventSetup& es);
  virtual void endRun(edm::Run &run, const edm::EventSetup& es);

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:

  void store(edm::Event& ev, const pixeltrackfitting::TracksWithRecHits & selectedTracks);

  edm::ParameterSet theConfig;

  const PixelFitter       * theFitter;
  const PixelTrackFilter  * theFilter;
        PixelTrackCleaner * theCleaner;  
        OrderedHitsGenerator * theGenerator;
        TrackingRegionProducer* theRegionProducer;
};
#endif
