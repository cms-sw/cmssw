#ifndef FastSimulation_Tracking_PixelTracksProducer_H
#define FastSimulation_Tracking_PixelTracksProducer_H

#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <string>

class PixelFitter;
class PixelTrackFilter;
class TrackingRegionProducer;

class PixelTracksProducer : public edm::stream::EDProducer<> {
public:
  explicit PixelTracksProducer(const edm::ParameterSet& conf);

  ~PixelTracksProducer() override;

  void produce(edm::Event& ev, const edm::EventSetup& es) override;

private:
  edm::EDGetTokenT<PixelFitter> fitterToken;
  std::unique_ptr<TrackingRegionProducer> theRegionProducer;

  edm::EDGetTokenT<TrajectorySeedCollection> seedProducerToken;
  edm::EDGetTokenT<PixelTrackFilter> filterToken;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken;
};
#endif
