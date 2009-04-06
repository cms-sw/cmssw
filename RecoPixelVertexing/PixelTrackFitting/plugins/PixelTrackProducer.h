#ifndef PixelTrackProducer_H
#define PixelTrackProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackReconstruction.h"

namespace edm { class Event; class EventSetup; class ParameterSet; }

class PixelTrackProducer :  public edm::EDProducer {

public:
  explicit PixelTrackProducer(const edm::ParameterSet& conf);

  ~PixelTrackProducer();

  virtual void beginRun(edm::Run &run, const edm::EventSetup& es);
  virtual void endRun(edm::Run &run, const edm::EventSetup& es);
  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:
  void store(edm::Event& ev, const pixeltrackfitting::TracksWithRecHits & selectedTracks);
  PixelTrackReconstruction theReconstruction;
};
#endif
