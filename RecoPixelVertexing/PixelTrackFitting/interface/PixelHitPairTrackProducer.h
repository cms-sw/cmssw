#ifndef PixelHitPairTrackProducer_H
#define PixelHitPairTrackProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PixelFitter;
class PixelTrackFilter;
class PixelSeedLayerPairs;

#include<vector>

class PixelHitPairTrackProducer :  public edm::EDProducer {

public:
  explicit PixelHitPairTrackProducer(const edm::ParameterSet& conf);

  ~PixelHitPairTrackProducer();

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:

  edm::ParameterSet theConfig;

  const PixelFitter      * theFitter;
  const PixelTrackFilter * theFilter;
  PixelSeedLayerPairs * pixelLayers;

};
#endif
