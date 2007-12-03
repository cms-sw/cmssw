#ifndef FastSimulation_Tracking_PixelTracksProducer_H
#define FastSimulation_Tracking_PixelTracksProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include <string>

class PixelFitter;
class PixelTrackFilter;
class TrackingRegionProducer;
class TrackerGeometry;

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}


class PixelTracksProducer :  public edm::EDProducer {

public:
  explicit PixelTracksProducer(const edm::ParameterSet& conf);

  ~PixelTracksProducer();

  virtual void beginJob(const edm::EventSetup& es);

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:

  const TrackerGeometry*  theGeometry;

  const PixelFitter       * theFitter;
  const PixelTrackFilter  * theFilter;
  TrackingRegionProducer* theRegionProducer;

  edm::InputTag seedProducer;

};
#endif


