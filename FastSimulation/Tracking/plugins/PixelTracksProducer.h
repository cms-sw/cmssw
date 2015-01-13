#ifndef FastSimulation_Tracking_PixelTracksProducer_H
#define FastSimulation_Tracking_PixelTracksProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include <string>

class PixelFitter;
class PixelTrackFilter;
class TrackingRegionProducer;

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}


class PixelTracksProducer :  public edm::stream::EDProducer <> {

public:
  explicit PixelTracksProducer(const edm::ParameterSet& conf);

  ~PixelTracksProducer();

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:

  const PixelFitter       * theFitter;
  const PixelTrackFilter  * theFilter;
  TrackingRegionProducer* theRegionProducer;

  edm::InputTag seedProducer;
  edm::EDGetTokenT<TrajectorySeedCollection> seedProducerToken;

};
#endif


