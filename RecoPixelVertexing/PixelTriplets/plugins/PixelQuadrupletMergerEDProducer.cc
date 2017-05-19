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
#include "RecoPixelVertexing/PixelTriplets/interface/QuadrupletSeedMerger.h"

// following are needed only to keep the same results
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"

class PixelQuadrupletMergerEDProducer: public edm::stream::EDProducer<> {
public:
  PixelQuadrupletMergerEDProducer(const edm::ParameterSet& iConfig);
  ~PixelQuadrupletMergerEDProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<RegionsSeedingHitSets> tripletToken_;

  edm::RunningAverage localRA_;

  QuadrupletSeedMerger merger_;

  // to keep old results
  std::unique_ptr<SeedComparitor> comparitor_;
  std::unique_ptr<SeedCreator> seedCreator_;
};

PixelQuadrupletMergerEDProducer::PixelQuadrupletMergerEDProducer(const edm::ParameterSet& iConfig):
  tripletToken_(consumes<RegionsSeedingHitSets>(iConfig.getParameter<edm::InputTag>("triplets"))),
  merger_(iConfig.getParameter<edm::ParameterSet>("layerList"), consumesCollector())
{
  merger_.setTTRHBuilderLabel(iConfig.getParameter<std::string>("ttrhBuilderLabel"));
  merger_.setMergeTriplets(iConfig.getParameter<bool>("mergeTriplets"));
  merger_.setAddRemainingTriplets(iConfig.getParameter<bool>("addRemainingTriplets"));

  edm::ParameterSet comparitorPSet = iConfig.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  if(comparitorName != "none") {
    auto iC = consumesCollector();
    comparitor_.reset(SeedComparitorFactory::get()->create(comparitorName, comparitorPSet, iC));
  }

  edm::ParameterSet creatorPSet = iConfig.getParameter<edm::ParameterSet>("SeedCreatorPSet");
  std::string creatorName = creatorPSet.getParameter<std::string>("ComponentName");
  if(creatorName != "none") // pixel tracking does not use seed creator
    seedCreator_.reset(SeedCreatorFactory::get()->create( creatorName, creatorPSet));

  produces<RegionsSeedingHitSets>();
  produces<TrajectorySeedCollection>(); // need to keep these in memory because TrajectorySeed owns its RecHits
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

  // to keep old results
  edm::ParameterSetDescription descComparitor;
  descComparitor.add<std::string>("ComponentName", "none");
  descComparitor.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("SeedComparitorPSet", descComparitor);
  edm::ParameterSetDescription descCreator;
  descCreator.add<std::string>("ComponentName", "none");
  descCreator.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("SeedCreatorPSet", descCreator);

  descriptions.add("pixelQuadrupletMergerEDProducer", desc);
}

void PixelQuadrupletMergerEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<RegionsSeedingHitSets> htriplets;
  iEvent.getByToken(tripletToken_, htriplets);
  const auto& regionTriplets = *htriplets;

  auto seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  if(regionTriplets.empty()) {
    iEvent.put(std::move(seedingHitSets));
    return;
  }
  seedingHitSets->reserve(regionTriplets.regionSize(), localRA_.upper());

  // to keep old results
  auto tmpSeedCollection = std::make_unique<TrajectorySeedCollection>();

  OrderedHitSeeds quadruplets;
  quadruplets.reserve(localRA_.upper());

  OrderedHitSeeds tripletsPerRegion;
  tripletsPerRegion.reserve(localRA_.upper());

  LogDebug("PixelQuadrupletMergerEDProducer") << "Creating quadruplets for " << regionTriplets.regionSize() << " regions from " << regionTriplets.size() << " triplets";
  merger_.update(iSetup);

  // to keep old results
  if(comparitor_) comparitor_->init(iEvent, iSetup);

  for(const auto& regionSeedingHitSets: regionTriplets) {
    const TrackingRegion& region = regionSeedingHitSets.region();
    auto seedingHitSetsFiller = seedingHitSets->beginRegion(&region);


    // Keeping same resuls has been made really difficult...
    // Especially when supporting both pixel tracking and seeding
    // Following is from SeedGeneratorFromRegionHits
    if(seedCreator_) {
      seedCreator_->init(region, iSetup, comparitor_.get());
      for(const auto& hits: regionSeedingHitSets) {
        if(!comparitor_ || comparitor_->compatible(hits)) {
          seedCreator_->makeSeed(*tmpSeedCollection, hits);
        }
      }

      // then convert seeds back to hits
      // awful, but hopefully only temporary to preserve old results
    for(const auto& seed: *tmpSeedCollection) {
        auto hitRange = seed.recHits();
        assert(std::distance(hitRange.first, hitRange.second) == 3);
        tripletsPerRegion.emplace_back(static_cast<SeedingHitSet::ConstRecHitPointer>(&*(hitRange.first)),
                                       static_cast<SeedingHitSet::ConstRecHitPointer>(&*(hitRange.first+1)),
                                       static_cast<SeedingHitSet::ConstRecHitPointer>(&*(hitRange.first+2)));
      }
    }
    else {
      for(const auto& hits: regionSeedingHitSets) {
        tripletsPerRegion.emplace_back(hits[0], hits[1], hits[2]);
      }
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

  seedingHitSets->shrink_to_fit();
  tmpSeedCollection->shrink_to_fit();
  iEvent.put(std::move(seedingHitSets));
  iEvent.put(std::move(tmpSeedCollection));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelQuadrupletMergerEDProducer);
