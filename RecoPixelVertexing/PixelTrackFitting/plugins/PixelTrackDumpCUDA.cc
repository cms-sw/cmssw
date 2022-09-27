#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

template <typename TrackerTraits>
class PixelTrackDumpCUDAT : public edm::global::EDAnalyzer<> {
public:
  using PixelTrackHeterogeneous = PixelTrackHeterogeneousT<TrackerTraits>;
  explicit PixelTrackDumpCUDAT(const edm::ParameterSet& iConfig);
  ~PixelTrackDumpCUDAT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const override;
  const bool m_onGPU;
  edm::EDGetTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenGPUTrack_;
  edm::EDGetTokenT<cms::cuda::Product<ZVertexHeterogeneous>> tokenGPUVertex_;
  edm::EDGetTokenT<PixelTrackHeterogeneous> tokenSoATrack_;
  edm::EDGetTokenT<ZVertexHeterogeneous> tokenSoAVertex_;
};

template <typename TrackerTraits>
PixelTrackDumpCUDAT<TrackerTraits>::PixelTrackDumpCUDAT(const edm::ParameterSet& iConfig)
    : m_onGPU(iConfig.getParameter<bool>("onGPU")) {
  if (m_onGPU) {
    tokenGPUTrack_ =
        consumes<cms::cuda::Product<PixelTrackHeterogeneous>>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenGPUVertex_ =
        consumes<cms::cuda::Product<ZVertexHeterogeneous>>(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"));
  } else {
    tokenSoATrack_ = consumes(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenSoAVertex_ = consumes<ZVertexHeterogeneous>(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"));
  }
}

template <typename TrackerTraits>
void PixelTrackDumpCUDAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("onGPU", true);
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("pixelTracksCUDA"));
  desc.add<edm::InputTag>("pixelVertexSrc", edm::InputTag("pixelVerticesCUDA"));
  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void PixelTrackDumpCUDAT<TrackerTraits>::analyze(edm::StreamID streamID,
                                                 edm::Event const& iEvent,
                                                 const edm::EventSetup& iSetup) const {
  if (m_onGPU) {
    auto const& hTracks = iEvent.get(tokenGPUTrack_);
    cms::cuda::ScopedContextProduce ctx{hTracks};

    auto const& tracks = ctx.get(hTracks);
    auto const* tsoa = tracks.get();
    assert(tsoa);

    auto const& vertices = ctx.get(iEvent.get(tokenGPUVertex_));
    auto const* vsoa = vertices.get();
    assert(vsoa);

  } else {
    auto const* tsoa = iEvent.get(tokenSoATrack_).get();
    assert(tsoa);

    auto const* vsoa = iEvent.get(tokenSoAVertex_).get();
    assert(vsoa);
  }
}

using PixelTrackDumpCUDA = PixelTrackDumpCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(PixelTrackDumpCUDA);

using PixelTrackDumpCUDAPhase1 = PixelTrackDumpCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(PixelTrackDumpCUDAPhase1);

using PixelTrackDumpCUDAPhase2 = PixelTrackDumpCUDAT<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(PixelTrackDumpCUDAPhase2);
