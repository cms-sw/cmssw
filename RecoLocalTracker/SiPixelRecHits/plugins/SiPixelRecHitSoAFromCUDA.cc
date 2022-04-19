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

class SiPixelRecHitSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelRecHitSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  using HMSstorage = HostProduct<uint32_t[]>;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  const edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DGPU>> hitsTokenGPU_;  // CUDA hits
  const edm::EDPutTokenT<TrackingRecHit2DCPU> hitsPutTokenCPU_;
  const edm::EDPutTokenT<HMSstorage> hostPutToken_;

  uint32_t nHits_;
  uint32_t nMaxModules_;

  cms::cuda::host::unique_ptr<float[]> store32_;
  cms::cuda::host::unique_ptr<uint16_t[]> store16_;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStart_;
};

SiPixelRecHitSoAFromCUDA::SiPixelRecHitSoAFromCUDA(const edm::ParameterSet& iConfig)
    : hitsTokenGPU_(
          consumes<cms::cuda::Product<TrackingRecHit2DGPU>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      hitsPutTokenCPU_(produces<TrackingRecHit2DCPU>()),
      hostPutToken_(produces<HMSstorage>()) {}

void SiPixelRecHitSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingCUDA"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelRecHitSoAFromCUDA::acquire(edm::Event const& iEvent,
                                       edm::EventSetup const& iSetup,
                                       edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<TrackingRecHit2DGPU> const& inputDataWrapped = iEvent.get(hitsTokenGPU_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  nHits_ = inputData.nHits();
  LogDebug("SiPixelRecHitSoAFromCUDA") << "copying to cpu SoA" << inputData.nHits() << " Hits";

  if (0 == nHits_)
    return;
  nMaxModules_ = inputData.nMaxModules();
  store32_ = inputData.store32ToHostAsync(ctx.stream());
  store16_ = inputData.store16ToHostAsync(ctx.stream());
  hitsModuleStart_ = inputData.hitsModuleStartToHostAsync(ctx.stream());
}

void SiPixelRecHitSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& es) {
  auto hmsp = std::make_unique<uint32_t[]>(nMaxModules_ + 1);
  std::copy(hitsModuleStart_.get(), hitsModuleStart_.get() + nMaxModules_ + 1, hmsp.get());

  iEvent.emplace(hostPutToken_, std::move(hmsp));
  iEvent.emplace(hitsPutTokenCPU_, store32_.get(), store16_.get(), hitsModuleStart_.get(), nHits_);
}

DEFINE_FWK_MODULE(SiPixelRecHitSoAFromCUDA);
