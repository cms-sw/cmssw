#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
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
#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

// Switch on to enable checks and printout for found tracks
// #define PIXEL_DEBUG_PRODUCE

class PixelTrackSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~PixelTrackSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> tokenSOA_;

  PixelTrackHeterogeneous soa_;
};

PixelTrackSoAFromCUDA::PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig)
    : tokenCUDA_(consumes<cms::cuda::Product<PixelTrackHeterogeneous>>(iConfig.getParameter<edm::InputTag>("src"))),
      tokenSOA_(produces<PixelTrackHeterogeneous>()) {}

void PixelTrackSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksCUDA"));
  descriptions.add("pixelTracksSoA", desc);
}

void PixelTrackSoAFromCUDA::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<PixelTrackHeterogeneous> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  soa_ = memoryPool::cuda::makeBuffer<PixelTrackHeterogeneous::value_type>(1, ctx.stream(), memoryPool::onHost);
  memoryPool::cuda::copy(soa_, inputData, 1, ctx.stream());
}

void PixelTrackSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
#ifdef PIXEL_DEBUG_PRODUCE
  auto const& tsoa = *soa_;
  auto maxTracks = tsoa.stride();
  auto nTracks = tsoa.nTracks();
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

  assert(!soa_.get());
}

DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);
