#include <cuda_runtime.h>

#include <fmt/printf.h>

#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

template <typename TrackerTraits>
class SiPixelRecHitSoAFromCUDAT : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelRecHitSoAFromCUDAT(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitSoAFromCUDAT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  using HMSstorage = HostProduct<uint32_t[]>;
  using HitsOnHost = TrackingRecHitSoAHost<TrackerTraits>;
  using HitsOnDevice = TrackingRecHitSoADevice<TrackerTraits>;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  const edm::EDGetTokenT<cms::cuda::Product<HitsOnDevice>> hitsTokenGPU_;  // CUDA hits
  const edm::EDPutTokenT<HitsOnHost> hitsPutTokenCPU_;
  const edm::EDPutTokenT<HMSstorage> hostPutToken_;

  uint32_t nHits_;
  HitsOnHost hits_h_;
};

template <typename TrackerTraits>
SiPixelRecHitSoAFromCUDAT<TrackerTraits>::SiPixelRecHitSoAFromCUDAT(const edm::ParameterSet& iConfig)
    : hitsTokenGPU_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      hitsPutTokenCPU_(produces<HitsOnHost>()),
      hostPutToken_(produces<HMSstorage>()) {}

template <typename TrackerTraits>
void SiPixelRecHitSoAFromCUDAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingCUDA"));

  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void SiPixelRecHitSoAFromCUDAT<TrackerTraits>::acquire(edm::Event const& iEvent,
                                                       edm::EventSetup const& iSetup,
                                                       edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<HitsOnDevice> const& inputDataWrapped = iEvent.get(hitsTokenGPU_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  nHits_ = inputData.nHits();
  hits_h_ = HitsOnHost(nHits_, ctx.stream());
  cudaCheck(cudaMemcpyAsync(hits_h_.buffer().get(),
                            inputData.const_buffer().get(),
                            inputData.bufferSize(),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));  // Copy data from Device to Host
  LogDebug("SiPixelRecHitSoAFromCUDA") << "copying to cpu SoA" << inputData.nHits() << " Hits";
}

template <typename TrackerTraits>
void SiPixelRecHitSoAFromCUDAT<TrackerTraits>::produce(edm::Event& iEvent, edm::EventSetup const& es) {
  auto hmsp = std::make_unique<uint32_t[]>(TrackerTraits::numberOfModules + 1);

  if (nHits_ > 0)
    std::copy(hits_h_.view().hitsModuleStart().begin(), hits_h_.view().hitsModuleStart().end(), hmsp.get());

  iEvent.emplace(hostPutToken_, std::move(hmsp));
  iEvent.emplace(hitsPutTokenCPU_, std::move(hits_h_));
}

using SiPixelRecHitSoAFromCUDA = SiPixelRecHitSoAFromCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiPixelRecHitSoAFromCUDA);

using SiPixelRecHitSoAFromCUDAPhase1 = SiPixelRecHitSoAFromCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiPixelRecHitSoAFromCUDAPhase1);

using SiPixelRecHitSoAFromCUDAPhase2 = SiPixelRecHitSoAFromCUDAT<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(SiPixelRecHitSoAFromCUDAPhase2);
