#ifndef RecoPixelVertexing_PixelTrackFitting_PixelTrackReconstruction_H
#define RecoPixelVertexing_PixelTrackFitting_PixelTrackReconstruction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

#include <memory>

class PixelFitter;
class PixelTrackCleaner;
class PixelTrackFilter;
class RegionsSeedingHitSets;

namespace edm {
  class Event;
  class EventSetup;
  class Run;
  class ParameterSetDescription;
}  // namespace edm

class PixelTrackReconstruction {
public:
  PixelTrackReconstruction(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);
  ~PixelTrackReconstruction();

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  void run(pixeltrackfitting::TracksWithTTRHs& tah, edm::Event& ev, const edm::EventSetup& es);

private:
  edm::EDGetTokenT<RegionsSeedingHitSets> theHitSetsToken;
  edm::EDGetTokenT<PixelFitter> theFitterToken;
  edm::EDGetTokenT<PixelTrackFilter> theFilterToken;
  std::string theCleanerName;
};
#endif
