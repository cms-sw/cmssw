#ifndef RecoPixelVertexing_PixelTrackFitting_PixelTrackReconstruction_H
#define RecoPixelVertexing_PixelTrackFitting_PixelTrackReconstruction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <memory>

class PixelFitter;
class PixelTrackCleaner;
class PixelTrackFilter;
class OrderedHitsGenerator;
class TrackingRegionProducer;
class QuadrupletSeedMerger;

namespace edm { class Event; class EventSetup; class Run; }

class PixelTrackReconstruction {
public:

  PixelTrackReconstruction( const edm::ParameterSet& conf,
	   edm::ConsumesCollector && iC);
  ~PixelTrackReconstruction(); 

  void run(pixeltrackfitting::TracksWithTTRHs& tah, edm::Event& ev, const edm::EventSetup& es);

  void halt();
  void init(const edm::EventSetup& es);

private:
  edm::ParameterSet theConfig;
  const PixelFitter       * theFitter;
  std::unique_ptr<PixelTrackFilter> theFilter;
  PixelTrackCleaner * theCleaner;
  std::unique_ptr<OrderedHitsGenerator> theGenerator;
  std::unique_ptr<TrackingRegionProducer> theRegionProducer;
  std::unique_ptr<QuadrupletSeedMerger> theMerger_;
};
#endif

