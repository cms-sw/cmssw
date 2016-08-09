#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/RunningAverage.h"

#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoPixelVertexing/PixelTriplets/interface/IntermediateHitTriplets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/QuadrupletSeedMerger.h"

class PixelQuadrupletMergerEDProducer: public edm::stream::EDProducer<> {
public:
  PixelQuadrupletMergerEDProducer(const edm::ParameterSet& iConfig);
  ~PixelQuadrupletMergerEDProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<IntermediateHitTriplets> tripletToken_;

  edm::RunningAverage localRA_;

  QuadrupletSeedMerger merger_;
};

PixelQuadrupletMergerEDProducer::PixelQuadrupletMergerEDProducer(const edm::ParameterSet& iConfig):
  tripletToken_(consumes<IntermediateHitTriplets>(iConfig.getParameter<edm::InputTag>("triplets"))),
  merger_(iConfig.getParameter<edm::ParameterSet>("layerList"), consumesCollector())
{
  merger_.setTTRHBuilderLabel(iConfig.getParameter<std::string>("ttrhBuilderLabel"));
  merger_.setMergeTriplets(iConfig.getParameter<bool>("mergeTriplets"));
  merger_.setAddRemainingTriplets(iConfig.getParameter<bool>("addRemainingTriplets"));

  produces<RegionsSeedingHitSets>();
}

void PixelQuadrupletMergerEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("triplets", edm::InputTag("hitTripletMergerEDProducer"));
  desc.add<std::string>("ttrhBuilderLabel", "PixelTTRHBuilderWithoutAngle");
  desc.add<bool>("mergeTriplets", true);
  desc.add<bool>("addRemainingTriplets", false);

  // This would be really on the responsibility of
  // QuadrupletSeedMerger and SeedingLayerSetsBuilder. The former is
  // almost obsolete by now (so I don't want to put effort there), and
  // the latter is better dealt in the context of SeedingLayersEDProducer.
  edm::ParameterSetDescription descLayers;
  descLayers.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("layerList", descLayers);

  descriptions.add("pixelQuadrupletMergerEDProducer", desc);
}

void PixelQuadrupletMergerEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<IntermediateHitTriplets> htriplets;
  iEvent.getByToken(tripletToken_, htriplets);
  const auto& regionTriplets = *htriplets;

  auto seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  if(regionTriplets.empty()) {
    iEvent.put(std::move(seedingHitSets));
    return;
  }
  seedingHitSets->reserve(regionTriplets.regionSize(), localRA_.upper());

  OrderedHitSeeds quadruplets;
  quadruplets.reserve(localRA_.upper());

  OrderedHitSeeds tripletsPerRegion;
  tripletsPerRegion.reserve(localRA_.upper());

  LogDebug("PixelQuadrupletMergerEDProducer") << "Creating quadruplets for " << regionTriplets.regionSize() << " regions from " << regionTriplets.tripletsSize() << " triplets";
  merger_.update(iSetup);

  for(const auto& regionLayerPairAndLayers: regionTriplets) {
    const TrackingRegion& region = regionLayerPairAndLayers.region();
    auto seedingHitSetsFiller = seedingHitSets->beginRegion(&region);


    for(const auto& layerTriplet: regionLayerPairAndLayers) {
      tripletsPerRegion.insert(tripletsPerRegion.end(), layerTriplet.tripletsBegin(), layerTriplet.tripletsEnd());
    }

    LogTrace("PixelQuadrupletEDProducer") << " starting region, number of triplets " << tripletsPerRegion.size();

    const auto& quadruplets = merger_.mergeTriplets(tripletsPerRegion, iSetup);

    LogTrace("PixelQuadrupletEDProducer") << " created " << quadruplets.size() << " quadruplets";

    for(size_t i=0; i!= quadruplets.size(); ++i) {
      const auto& quad = quadruplets[i];
      seedingHitSetsFiller.emplace_back(quad[0], quad[1], quad[2], quad[3]);
    }

    tripletsPerRegion.clear();
  }
  localRA_.update(seedingHitSets->size());

  iEvent.put(std::move(seedingHitSets));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelQuadrupletMergerEDProducer);
