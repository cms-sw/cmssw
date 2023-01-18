#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"

#include "gpuVertexFinder.h"

#undef PIXVERTEX_DEBUG_PRODUCE

template <typename TrackerTraits>
class PixelVertexProducerCUDAT : public edm::global::EDProducer<> {
  using TracksSoADevice = TrackSoAHeterogeneousDevice<TrackerTraits>;
  using TracksSoAHost = TrackSoAHeterogeneousHost<TrackerTraits>;
  using GPUAlgo = gpuVertexFinder::Producer<TrackerTraits>;

public:
  explicit PixelVertexProducerCUDAT(const edm::ParameterSet& iConfig);
  ~PixelVertexProducerCUDAT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produceOnGPU(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const;
  void produceOnCPU(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const;
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  bool onGPU_;

  edm::EDGetTokenT<cms::cuda::Product<TracksSoADevice>> tokenGPUTrack_;
  edm::EDPutTokenT<cms::cuda::Product<ZVertexSoADevice>> tokenGPUVertex_;
  edm::EDGetTokenT<TracksSoAHost> tokenCPUTrack_;
  edm::EDPutTokenT<ZVertexSoAHost> tokenCPUVertex_;

  const GPUAlgo gpuAlgo_;

  // Tracking cuts before sending tracks to vertex algo
  const float ptMin_;
  const float ptMax_;
};

template <typename TrackerTraits>
PixelVertexProducerCUDAT<TrackerTraits>::PixelVertexProducerCUDAT(const edm::ParameterSet& conf)
    : onGPU_(conf.getParameter<bool>("onGPU")),
      gpuAlgo_(conf.getParameter<bool>("oneKernel"),
               conf.getParameter<bool>("useDensity"),
               conf.getParameter<bool>("useDBSCAN"),
               conf.getParameter<bool>("useIterative"),
               conf.getParameter<int>("minT"),
               conf.getParameter<double>("eps"),
               conf.getParameter<double>("errmax"),
               conf.getParameter<double>("chi2max")),
      ptMin_(conf.getParameter<double>("PtMin")),  // 0.5 GeV
      ptMax_(conf.getParameter<double>("PtMax"))   // 75. GeV
{
  if (onGPU_) {
    tokenGPUTrack_ = consumes(conf.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenGPUVertex_ = produces();
  } else {
    tokenCPUTrack_ = consumes(conf.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenCPUVertex_ = produces();
  }
}

template <typename TrackerTraits>
void PixelVertexProducerCUDAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // Only one of these three algos can be used at once.
  // Maybe this should become a Plugin Factory
  desc.add<bool>("onGPU", true);
  desc.add<bool>("oneKernel", true);
  desc.add<bool>("useDensity", true);
  desc.add<bool>("useDBSCAN", false);
  desc.add<bool>("useIterative", false);

  desc.add<int>("minT", 2);          // min number of neighbours to be "core"
  desc.add<double>("eps", 0.07);     // max absolute distance to cluster
  desc.add<double>("errmax", 0.01);  // max error to be "seed"
  desc.add<double>("chi2max", 9.);   // max normalized distance to cluster

  desc.add<double>("PtMin", 0.5);
  desc.add<double>("PtMax", 75.);
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("pixelTracksCUDA"));

  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void PixelVertexProducerCUDAT<TrackerTraits>::produceOnGPU(edm::StreamID streamID,
                                                           edm::Event& iEvent,
                                                           const edm::EventSetup& iSetup) const {
  using TracksSoA = TrackSoAHeterogeneousDevice<TrackerTraits>;
  auto hTracks = iEvent.getHandle(tokenGPUTrack_);

  cms::cuda::ScopedContextProduce ctx{*hTracks};
  auto& tracks = ctx.get(*hTracks);

  ctx.emplace(iEvent, tokenGPUVertex_, gpuAlgo_.makeAsync(ctx.stream(), tracks.view(), ptMin_, ptMax_));
}

template <typename TrackerTraits>
void PixelVertexProducerCUDAT<TrackerTraits>::produceOnCPU(edm::StreamID streamID,
                                                           edm::Event& iEvent,
                                                           const edm::EventSetup& iSetup) const {
  auto& tracks = iEvent.get(tokenCPUTrack_);

#ifdef PIXVERTEX_DEBUG_PRODUCE
  auto const& tsoa = *tracks;
  auto maxTracks = tsoa.stride();
  std::cout << "size of SoA " << sizeof(tsoa) << " stride " << maxTracks << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = TracksUtilities<TrackerTraits>::nHits(tracks.view(), it);
    assert(nHits == int(tracks.view().hitIndices().size(it)));
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  std::cout << "found " << nt << " tracks in cpu SoA for Vertexing at " << tracks << std::endl;
#endif  // PIXVERTEX_DEBUG_PRODUCE

  iEvent.emplace(tokenCPUVertex_, gpuAlgo_.make(tracks.view(), ptMin_, ptMax_));
}

template <typename TrackerTraits>
void PixelVertexProducerCUDAT<TrackerTraits>::produce(edm::StreamID streamID,
                                                      edm::Event& iEvent,
                                                      const edm::EventSetup& iSetup) const {
  if (onGPU_) {
    produceOnGPU(streamID, iEvent, iSetup);
  } else {
    produceOnCPU(streamID, iEvent, iSetup);
  }
}

using PixelVertexProducerCUDA = PixelVertexProducerCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(PixelVertexProducerCUDA);

using PixelVertexProducerCUDAPhase1 = PixelVertexProducerCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(PixelVertexProducerCUDAPhase1);

using PixelVertexProducerCUDAPhase2 = PixelVertexProducerCUDAT<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(PixelVertexProducerCUDAPhase2);
