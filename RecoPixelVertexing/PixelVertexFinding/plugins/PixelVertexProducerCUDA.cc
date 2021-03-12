#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
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

#include "gpuVertexFinder.h"

class PixelVertexProducerCUDA : public edm::global::EDProducer<> {
public:
  explicit PixelVertexProducerCUDA(const edm::ParameterSet& iConfig);
  ~PixelVertexProducerCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  bool m_OnGPU;

  edm::EDGetTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenGPUTrack_;
  edm::EDPutTokenT<ZVertexCUDAProduct> tokenGPUVertex_;
  edm::EDGetTokenT<PixelTrackHeterogeneous> tokenCPUTrack_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenCPUVertex_;

  const gpuVertexFinder::Producer m_gpuAlgo;

  // Tracking cuts before sending tracks to vertex algo
  const float m_ptMin;
};

PixelVertexProducerCUDA::PixelVertexProducerCUDA(const edm::ParameterSet& conf)
    : m_OnGPU(conf.getParameter<bool>("onGPU")),
      m_gpuAlgo(conf.getParameter<bool>("oneKernel"),
                conf.getParameter<bool>("useDensity"),
                conf.getParameter<bool>("useDBSCAN"),
                conf.getParameter<bool>("useIterative"),
                conf.getParameter<int>("minT"),
                conf.getParameter<double>("eps"),
                conf.getParameter<double>("errmax"),
                conf.getParameter<double>("chi2max")),
      m_ptMin(conf.getParameter<double>("PtMin"))  // 0.5 GeV
{
  if (m_OnGPU) {
    tokenGPUTrack_ =
        consumes<cms::cuda::Product<PixelTrackHeterogeneous>>(conf.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenGPUVertex_ = produces<ZVertexCUDAProduct>();
  } else {
    tokenCPUTrack_ = consumes<PixelTrackHeterogeneous>(conf.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenCPUVertex_ = produces<ZVertexHeterogeneous>();
  }
}

void PixelVertexProducerCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("caHitNtupletCUDA"));

  auto label = "pixelVertexCUDA";
  descriptions.add(label, desc);
}

void PixelVertexProducerCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  if (m_OnGPU) {
    edm::Handle<cms::cuda::Product<PixelTrackHeterogeneous>> hTracks;
    iEvent.getByToken(tokenGPUTrack_, hTracks);

    cms::cuda::ScopedContextProduce ctx{*hTracks};
    auto const* tracks = ctx.get(*hTracks).get();

    assert(tracks);

    ctx.emplace(iEvent, tokenGPUVertex_, m_gpuAlgo.makeAsync(ctx.stream(), tracks, m_ptMin));

  } else {
    auto const* tracks = iEvent.get(tokenCPUTrack_).get();
    assert(tracks);

    /*
    auto const & tsoa = *tracks;
    auto maxTracks = tsoa.stride();
    std::cout << "size of SoA " << sizeof(tsoa) << " stride " << maxTracks << std::endl;

    int32_t nt = 0;
    for (int32_t it = 0; it < maxTracks; ++it) {
      auto nHits = tsoa.nHits(it);
      assert(nHits==int(tsoa.hitIndices.size(it)));
      if (nHits == 0) break;  // this is a guard: maybe we need to move to nTracks...
      nt++;
    }
    std::cout << "found " << nt << " tracks in cpu SoA for Vertexing at " << tracks << std::endl;
    */

    iEvent.emplace(tokenCPUVertex_, m_gpuAlgo.make(tracks, m_ptMin));
  }
}

DEFINE_FWK_MODULE(PixelVertexProducerCUDA);
