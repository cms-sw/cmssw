#include <cuda_runtime.h>

#include <fmt/printf.h>

#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
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
  using TrackingRecHit2DSOAView = TrackingRecHit2DSOAViewT<TrackerTraits>;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  const edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DGPUT<TrackerTraits>>> hitsTokenGPU_;  // CUDA hits
  const edm::EDPutTokenT<TrackingRecHit2DCPUT<TrackerTraits>> hitsPutTokenCPU_;
  const edm::EDPutTokenT<HMSstorage> hostPutToken_;

  uint32_t nHits_;

  cms::cuda::host::unique_ptr<float[]> store32_;
  cms::cuda::host::unique_ptr<uint16_t[]> store16_;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStart_;
};

template <typename TrackerTraits>
SiPixelRecHitSoAFromCUDAT<TrackerTraits>::SiPixelRecHitSoAFromCUDAT(const edm::ParameterSet& iConfig)
    : hitsTokenGPU_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      hitsPutTokenCPU_(produces<TrackingRecHit2DCPUT<TrackerTraits>>()),
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
  cms::cuda::Product<TrackingRecHit2DGPUT<TrackerTraits>> const& inputDataWrapped = iEvent.get(hitsTokenGPU_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  nHits_ = inputData.nHits();
  LogDebug("SiPixelRecHitSoAFromCUDA") << "copying to cpu SoA" << inputData.nHits() << " Hits";

  if (0 == nHits_)
    return;
  store32_ = inputData.store32ToHostAsync(ctx.stream());
  store16_ = inputData.store16ToHostAsync(ctx.stream());
  hitsModuleStart_ = inputData.hitsModuleStartToHostAsync(ctx.stream());
}

template <typename TrackerTraits>
void SiPixelRecHitSoAFromCUDAT<TrackerTraits>::produce(edm::Event& iEvent, edm::EventSetup const& es) {
  auto hmsp = std::make_unique<uint32_t[]>(TrackerTraits::numberOfModules + 1);

  if (nHits_ > 0)
    std::copy(hitsModuleStart_.get(), hitsModuleStart_.get() + TrackerTraits::numberOfModules + 1, hmsp.get());

  iEvent.emplace(hostPutToken_, std::move(hmsp));
  iEvent.emplace(hitsPutTokenCPU_, store32_, store16_, hitsModuleStart_.get(), nHits_);
}

using SiPixelRecHitSoAFromCUDA = SiPixelRecHitSoAFromCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiPixelRecHitSoAFromCUDA);

using SiPixelRecHitSoAFromCUDAPhase1 = SiPixelRecHitSoAFromCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiPixelRecHitSoAFromCUDAPhase1);

using SiPixelRecHitSoAFromCUDAPhase2 = SiPixelRecHitSoAFromCUDAT<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(SiPixelRecHitSoAFromCUDAPhase2);
