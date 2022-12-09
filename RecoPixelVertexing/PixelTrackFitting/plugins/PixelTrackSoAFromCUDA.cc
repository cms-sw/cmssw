#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

// Switch on to enable checks and printout for found tracks
// #define PIXEL_DEBUG_PRODUCE

template <typename TrackerTraits>
class PixelTrackSoAFromCUDAT : public edm::stream::EDProducer<edm::ExternalWork> {
  using PixelTrackHeterogeneous = PixelTrackHeterogeneousT<TrackerTraits>;
  using TrackSoA = pixelTrack::TrackSoAT<TrackerTraits>;

public:
  explicit PixelTrackSoAFromCUDAT(const edm::ParameterSet& iConfig);
  ~PixelTrackSoAFromCUDAT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> tokenSOA_;

  cms::cuda::host::unique_ptr<TrackSoA> soa_;
};

template <typename TrackerTraits>
PixelTrackSoAFromCUDAT<TrackerTraits>::PixelTrackSoAFromCUDAT(const edm::ParameterSet& iConfig)
    : tokenCUDA_(consumes<cms::cuda::Product<PixelTrackHeterogeneous>>(iConfig.getParameter<edm::InputTag>("src"))),
      tokenSOA_(produces<PixelTrackHeterogeneous>()) {}

template <typename TrackerTraits>
void PixelTrackSoAFromCUDAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksCUDA"));
  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void PixelTrackSoAFromCUDAT<TrackerTraits>::acquire(edm::Event const& iEvent,
                                                    edm::EventSetup const& iSetup,
                                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<PixelTrackHeterogeneous> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  soa_ = inputData.toHostAsync(ctx.stream());
}

template <typename TrackerTraits>
void PixelTrackSoAFromCUDAT<TrackerTraits>::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  auto const& tsoa = *soa_;
  auto maxTracks = tsoa.stride();

  auto nTracks = tsoa.nTracks();
  assert(nTracks < maxTracks);
  if (nTracks == maxTracks - 1) {
    edm::LogWarning("PixelTracks") << "Unsorted reconstructed pixel tracks truncated to " << maxTracks - 1
                                   << " candidates";
  }

#ifdef PIXEL_DEBUG_PRODUCE
  std::cout << "size of SoA " << sizeof(tsoa) << " stride " << maxTracks << std::endl;
  std::cout << "found " << nTracks << " tracks in cpu SoA at " << &tsoa << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    assert(nHits == int(tsoa.hitIndices.size(it)));
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  assert(nTracks == nt);
#endif

  // DO NOT  make a copy  (actually TWO....)
  iEvent.emplace(tokenSOA_, std::move(soa_));

  assert(!soa_);
}

using PixelTrackSoAFromCUDA = PixelTrackSoAFromCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);

using PixelTrackSoAFromCUDAPhase1 = PixelTrackSoAFromCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(PixelTrackSoAFromCUDAPhase1);

using PixelTrackSoAFromCUDAPhase2 = PixelTrackSoAFromCUDAT<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(PixelTrackSoAFromCUDAPhase2);
