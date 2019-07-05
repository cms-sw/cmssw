#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"


class CAHitNtupletCUDA : public edm::global::EDProducer<> {
public:
  explicit CAHitNtupletCUDA(const edm::ParameterSet& iConfig);
  ~CAHitNtupletCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<CUDAProduct<TrackingRecHit2DCUDA>> tokenHit_;
  edm::EDPutTokenT<CUDAProduct<PixelTrackHeterogeneous>> tokenTrack_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;

};

CAHitNtupletCUDA::CAHitNtupletCUDA(const edm::ParameterSet& iConfig) :
      tokenHit_(consumes<CUDAProduct<TrackingRecHit2DCUDA>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      tokenTrack_(produces<CUDAProduct<PixelTrackHeterogeneous>>()),
      gpuAlgo_(iConfig, consumesCollector()) {}


void CAHitNtupletCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsCUDAPreSplitting"));

  CAHitNtupletGeneratorOnGPU::fillDescriptions(desc);
  auto label = "caHitNtupletCUDA";
  descriptions.add(label, desc);
}

void CAHitNtupletCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& es) const {

  edm::Handle<CUDAProduct<TrackingRecHit2DCUDA>>  hHits;
  iEvent.getByToken(tokenHit_, hHits);

  CUDAScopedContextProduce ctx{*hHits};
  auto const& hits = ctx.get(*hHits);

  auto bf = 1./PixelRecoUtilities::fieldInInvGev(es);

  ctx.emplace(
      iEvent,
      tokenTrack_,
      std::move(gpuAlgo_.makeTuplesAsync(hits, bf, ctx.stream()))
   );

}



DEFINE_FWK_MODULE(CAHitNtupletCUDA);
