#ifndef PixelTrackProducer_H
#define PixelTrackProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"

class PixelFitter;
class PixelTrackCleaner;
class PixelTrackFilter;
class PixelHitTripletGenerator;

#include<vector>

class PixelTrackProducer :  public edm::EDProducer {

public:
  explicit PixelTrackProducer(const edm::ParameterSet& conf);

  ~PixelTrackProducer();

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:

  void store(edm::Event& ev, const pixeltrackfitting::TracksWithRecHits & selectedTracks);

  edm::ParameterSet theConfig;

  const PixelFitter       * theFitter;
  const PixelTrackFilter  * theFilter;
        PixelTrackCleaner * theCleaner;  
        PixelHitTripletGenerator * theGenerator;

};
#endif
