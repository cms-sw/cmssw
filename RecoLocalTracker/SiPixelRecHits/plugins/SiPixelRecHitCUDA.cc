#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

#include "PixelRecHitGPUKernel.h"

template <typename TrackerTraits>
class SiPixelRecHitCUDAT : public edm::global::EDProducer<> {
public:
  explicit SiPixelRecHitCUDAT(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitCUDAT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> cpeToken_;
  const edm::EDGetTokenT<cms::cuda::Product<BeamSpotCUDA>> tBeamSpot;
  const edm::EDGetTokenT<cms::cuda::Product<SiPixelClustersCUDA>> token_;
  const edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> tokenDigi_;
  const edm::EDPutTokenT<cms::cuda::Product<TrackingRecHitSoADevice<TrackerTraits>>> tokenHit_;

  const pixelgpudetails::PixelRecHitGPUKernel<TrackerTraits> gpuAlgo_;
};

template <typename TrackerTraits>
SiPixelRecHitCUDAT<TrackerTraits>::SiPixelRecHitCUDAT(const edm::ParameterSet& iConfig)
    : cpeToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("CPE")))),
      tBeamSpot(consumes<cms::cuda::Product<BeamSpotCUDA>>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      token_(consumes<cms::cuda::Product<SiPixelClustersCUDA>>(iConfig.getParameter<edm::InputTag>("src"))),
      tokenDigi_(consumes<cms::cuda::Product<SiPixelDigisCUDA>>(iConfig.getParameter<edm::InputTag>("src"))),
      tokenHit_(produces<cms::cuda::Product<TrackingRecHitSoADevice<TrackerTraits>>>()) {}

template <typename TrackerTraits>
void SiPixelRecHitCUDAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpotCUDA"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplittingCUDA"));

  std::string cpe = "PixelCPEFast";
  cpe += TrackerTraits::nameModifier;
  desc.add<std::string>("CPE", cpe);

  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void SiPixelRecHitCUDAT<TrackerTraits>::produce(edm::StreamID streamID,
                                                edm::Event& iEvent,
                                                const edm::EventSetup& es) const {
  PixelCPEFast<TrackerTraits> const* fcpe = dynamic_cast<const PixelCPEFast<TrackerTraits>*>(&es.getData(cpeToken_));
  if (not fcpe) {
    throw cms::Exception("Configuration") << "SiPixelRecHitCUDA can only use a CPE of type PixelCPEFast";
  }

  edm::Handle<cms::cuda::Product<SiPixelClustersCUDA>> hclusters;
  iEvent.getByToken(token_, hclusters);

  cms::cuda::ScopedContextProduce ctx{*hclusters};
  auto const& clusters = ctx.get(*hclusters);

  edm::Handle<cms::cuda::Product<SiPixelDigisCUDA>> hdigis;
  iEvent.getByToken(tokenDigi_, hdigis);
  auto const& digis = ctx.get(*hdigis);

  edm::Handle<cms::cuda::Product<BeamSpotCUDA>> hbs;
  iEvent.getByToken(tBeamSpot, hbs);
  auto const& bs = ctx.get(*hbs);

  ctx.emplace(iEvent,
              tokenHit_,
              gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe->getGPUProductAsync(ctx.stream()), ctx.stream()));
}

using SiPixelRecHitCUDAPhase1 = SiPixelRecHitCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiPixelRecHitCUDAPhase1);

using SiPixelRecHitCUDAPhase2 = SiPixelRecHitCUDAT<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(SiPixelRecHitCUDAPhase2);

using SiPixelRecHitCUDAHIonPhase1 = SiPixelRecHitCUDAT<pixelTopology::HIonPhase1>;
DEFINE_FWK_MODULE(SiPixelRecHitCUDAHIonPhase1);
