#ifndef RecoPixelVertexing_PixelTrackFitting_plugins_PixelTrackReconstructionGPU_h
#define RecoPixelVertexing_PixelTrackFitting_plugins_PixelTrackReconstructionGPU_h

#include <memory>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"

class PixelFitter;
class PixelTrackCleaner;
class PixelTrackFilter;
class RegionsSeedingHitSets;

namespace edm {
  class Event;
  class EventSetup;
  class Run;
  class ParameterSetDescription;
}

class PixelTrackReconstructionGPU {
public:

  PixelTrackReconstructionGPU( const edm::ParameterSet& conf, edm::ConsumesCollector && iC);
  ~PixelTrackReconstructionGPU();

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  void run(pixeltrackfitting::TracksWithTTRHs& tah, edm::Event& ev, const edm::EventSetup& es);
  void launchKernelFit(float * hits_and_covariances, int cumulative_size, int hits_in_fit, float B,
      Rfit::helix_fit * results);

private:
  edm::EDGetTokenT<RegionsSeedingHitSets> theHitSetsToken;
  edm::EDGetTokenT<PixelFitter> theFitterToken;
  edm::EDGetTokenT<PixelTrackFilter> theFilterToken;
  std::string theCleanerName;
};

#endif // RecoPixelVertexing_PixelTrackFitting_plugins_PixelTrackReconstructionGPU_h
