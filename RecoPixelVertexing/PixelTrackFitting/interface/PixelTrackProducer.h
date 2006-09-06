#ifndef PixelTrackProducer_H
#define PixelTrackProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"

#include<vector>

class PixelTrackProducer :  public edm::EDProducer {

public:
  explicit PixelTrackProducer(const edm::ParameterSet& conf);

  ~PixelTrackProducer();

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);
  void buildTracks(edm::Event& ev, const edm::EventSetup& es);
  void filterTracks(edm::Event& ev, const edm::EventSetup& es);
  void addTracks(edm::Event& ev, const edm::EventSetup& es);

private:

  edm::ParameterSet theConfig;
  std::vector <PixelTrackCleaner::TrackHitsPair> allTracks, cleanedTracks;

};
#endif
