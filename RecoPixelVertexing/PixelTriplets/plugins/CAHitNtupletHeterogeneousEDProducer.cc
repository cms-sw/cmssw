#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CAHitQuadrupletGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"

#include "CAHitQuadrupletGeneratorGPU.h"

namespace {
void fillNtuplets(RegionsSeedingHitSets::RegionFiller &seedingHitSetsFiller,
                  const OrderedHitSeeds &quadruplets) {
  for (const auto &quad : quadruplets) {
    seedingHitSetsFiller.emplace_back(quad[0], quad[1], quad[2], quad[3]);
  }
}
} // namespace

class CAHitNtupletHeterogeneousEDProducer
    : public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<
          heterogeneous::GPUCuda, heterogeneous::CPU>> {
public:

  using PixelRecHitsH = siPixelRecHitsHeterogeneousProduct::HeterogeneousPixelRecHit;


  CAHitNtupletHeterogeneousEDProducer(const edm::ParameterSet &iConfig);
  ~CAHitNtupletHeterogeneousEDProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  void beginStreamGPUCuda(edm::StreamID streamId,
                          cuda::stream_t<> &cudaStream) override;
  void acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceCPU(edm::HeterogeneousEvent &iEvent,
                  const edm::EventSetup &iSetup) override;

private:
  edm::EDGetTokenT<IntermediateHitDoublets> doubletToken_;

  edm::EDGetTokenT<HeterogeneousProduct> tGpuHits;


  edm::RunningAverage localRA_;
  CAHitQuadrupletGeneratorGPU GPUGenerator_;
  CAHitQuadrupletGenerator CPUGenerator_;

  bool emptyRegionDoublets = false;
  std::unique_ptr<RegionsSeedingHitSets> seedingHitSets_;
};

CAHitNtupletHeterogeneousEDProducer::CAHitNtupletHeterogeneousEDProducer(
    const edm::ParameterSet &iConfig)
    : HeterogeneousEDProducer(iConfig),
      doubletToken_(consumes<IntermediateHitDoublets>(
          iConfig.getParameter<edm::InputTag>("doublets"))),
      tGpuHits(consumesHeterogeneous(iConfig.getParameter<edm::InputTag>("heterogeneousPixelRecHitSrc"))),
      GPUGenerator_(iConfig, consumesCollector()),
      CPUGenerator_(iConfig, consumesCollector()) {
  produces<RegionsSeedingHitSets>();
}

void CAHitNtupletHeterogeneousEDProducer::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("doublets", edm::InputTag("hitPairEDProducer"));

  desc.add<edm::InputTag>("heterogeneousPixelRecHitSrc", edm::InputTag("siPixelRecHitHeterogeneous"));

  CAHitQuadrupletGeneratorGPU::fillDescriptions(desc);
  HeterogeneousEDProducer::fillPSetDescription(desc);
  auto label = "caHitQuadrupletHeterogeneousEDProducer";
  descriptions.add(label, desc);
}

void CAHitNtupletHeterogeneousEDProducer::beginStreamGPUCuda(
    edm::StreamID streamId, cuda::stream_t<> &cudaStream) {
  GPUGenerator_.allocateOnGPU();
}

void CAHitNtupletHeterogeneousEDProducer::acquireGPUCuda(
    const edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup,
    cuda::stream_t<> &cudaStream) {

  seedingHitSets_ = std::make_unique<RegionsSeedingHitSets>();


  // FIXME: move directly to region or similar...
  edm::Handle<IntermediateHitDoublets> hdoublets;
  iEvent.getByToken(doubletToken_, hdoublets);
  const auto &regionDoublets = *hdoublets;
  assert(regionDoublets.regionSize()<=1);

  if (regionDoublets.empty()) {
    emptyRegionDoublets = true;
    return;
  }

  const TrackingRegion &region = (*regionDoublets.begin()).region();


  edm::Handle<siPixelRecHitsHeterogeneousProduct::GPUProduct> gh;
  iEvent.getByToken<siPixelRecHitsHeterogeneousProduct::HeterogeneousPixelRecHit>(tGpuHits, gh);
  auto const & gHits = *gh;
//  auto nhits = gHits.nHits;

  // move inside hitNtuplets???
  GPUGenerator_.buildDoublets(gHits,cudaStream.id());

  seedingHitSets_->reserve(regionDoublets.regionSize(), localRA_.upper());
  GPUGenerator_.initEvent(iEvent.event(), iSetup);

  LogDebug("CAHitNtupletHeterogeneousEDProducer")
        << "Creating ntuplets for " << regionDoublets.regionSize()
        << " regions, and " << regionDoublets.layerPairsSize()
        << " layer pairs";

  GPUGenerator_.hitNtuplets(region, gHits, iSetup, cudaStream.id());
  
}

void CAHitNtupletHeterogeneousEDProducer::produceGPUCuda(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup,
    cuda::stream_t<> &cudaStream) {

  if (not emptyRegionDoublets) {
    edm::Handle<IntermediateHitDoublets> hdoublets;
    iEvent.getByToken(doubletToken_, hdoublets);
    const auto &regionDoublets = *hdoublets;

    edm::Handle<HeterogeneousProduct> gh;
    iEvent.getByToken(tGpuHits, gh);
    auto const & rechits = gh->get<siPixelRecHitsHeterogeneousProduct::HeterogeneousPixelRecHit>().getProduct<HeterogeneousDevice::kCPU>();

    std::vector<OrderedHitSeeds> ntuplets(regionDoublets.regionSize());
    for (auto &ntuplet : ntuplets) ntuplet.reserve(localRA_.upper());
    int index = 0;
    for (const auto &regionLayerPairs : regionDoublets) {
      const TrackingRegion &region = regionLayerPairs.region();
      auto seedingHitSetsFiller = seedingHitSets_->beginRegion(&region);
      GPUGenerator_.fillResults(region, rechits.collection, ntuplets, iSetup, cudaStream.id());
      fillNtuplets(seedingHitSetsFiller, ntuplets[index]);
      ntuplets[index].clear();
      index++;
    }
    localRA_.update(seedingHitSets_->size());
  }
  iEvent.put(std::move(seedingHitSets_));
}

void CAHitNtupletHeterogeneousEDProducer::produceCPU(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<IntermediateHitDoublets> hdoublets;
  iEvent.getByToken(doubletToken_, hdoublets);
  const auto &regionDoublets = *hdoublets;

  const SeedingLayerSetsHits &seedingLayerHits = regionDoublets.seedingLayerHits();
  if (seedingLayerHits.numberOfLayersInSet() < CAHitQuadrupletGenerator::minLayers) {
    throw cms::Exception("LogicError")
        << "CAHitNtupletEDProducer expects "
           "SeedingLayerSetsHits::numberOfLayersInSet() to be >= "
        << CAHitQuadrupletGenerator::minLayers << ", got "
        << seedingLayerHits.numberOfLayersInSet()
        << ". This is likely caused by a configuration error of this module, "
           "HitPairEDProducer, or SeedingLayersEDProducer.";
  }

  auto seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  if (regionDoublets.empty()) {
    iEvent.put(std::move(seedingHitSets));
    return;
  }

  seedingHitSets->reserve(regionDoublets.regionSize(), localRA_.upper());
  CPUGenerator_.initEvent(iEvent.event(), iSetup);

  LogDebug("CAHitNtupletEDProducer")
      << "Creating ntuplets for " << regionDoublets.regionSize()
      << " regions, and " << regionDoublets.layerPairsSize() << " layer pairs";
  std::vector<OrderedHitSeeds> ntuplets;
  ntuplets.resize(regionDoublets.regionSize());
  for (auto &ntuplet : ntuplets)
    ntuplet.reserve(localRA_.upper());

  CPUGenerator_.hitNtuplets(regionDoublets, ntuplets, iSetup, seedingLayerHits);
  int index = 0;
  for (const auto &regionLayerPairs : regionDoublets) {
    const TrackingRegion &region = regionLayerPairs.region();
    auto seedingHitSetsFiller = seedingHitSets->beginRegion(&region);

    fillNtuplets(seedingHitSetsFiller, ntuplets[index]);
    ntuplets[index].clear();
    index++;
  }
  localRA_.update(seedingHitSets->size());

  iEvent.put(std::move(seedingHitSets));
}

DEFINE_FWK_MODULE(CAHitNtupletHeterogeneousEDProducer);
