#ifndef PixelTrackProducer_H
#define PixelTrackProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackReconstruction.h"

namespace edm { class Event; class EventSetup; class ParameterSet; }

class PixelTrackProducer :  public edm::stream::EDProducer<> {

public:
  explicit PixelTrackProducer(const edm::ParameterSet& conf);

  ~PixelTrackProducer();

  virtual void beginRun(const edm::Run &run, const edm::EventSetup& es) override;
  virtual void endRun(const edm::Run &run, const edm::EventSetup& es) override;
  virtual void produce(edm::Event& ev, const edm::EventSetup& es) override;

private:
  void store(edm::Event& ev, const pixeltrackfitting::TracksWithTTRHs& selectedTracks);
  PixelTrackReconstruction theReconstruction;
};
#endif
