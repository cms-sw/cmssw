#include "RecoMuon/TrackerSeedGenerator/plugins/TSGSmart.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"

TSGSmart::TSGSmart(const edm::ParameterSet &pset, edm::ConsumesCollector &iC) {
  theEtaBound = pset.getParameter<double>("EtaBound");

  // FIXME??
  edm::ParameterSet creatorPSet;
  creatorPSet.addParameter<std::string>("propagator", "PropagatorWithMaterial");

  edm::ParameterSet PairPSet = pset.getParameter<edm::ParameterSet>("PixelPairGeneratorSet");
  edm::ParameterSet pairhitsfactoryPSet = PairPSet.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string pairhitsfactoryName = pairhitsfactoryPSet.getParameter<std::string>("ComponentName");

  thePairGenerator = std::make_unique<SeedGeneratorFromRegionHits>(
      OrderedHitsGeneratorFactory::get()->create(pairhitsfactoryName, pairhitsfactoryPSet, iC),
      nullptr,
      SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", creatorPSet));

  edm::ParameterSet TripletPSet = pset.getParameter<edm::ParameterSet>("PixelTripletGeneratorSet");
  edm::ParameterSet triplethitsfactoryPSet = TripletPSet.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string triplethitsfactoryName = triplethitsfactoryPSet.getParameter<std::string>("ComponentName");

  theTripletGenerator = std::make_unique<SeedGeneratorFromRegionHits>(
      OrderedHitsGeneratorFactory::get()->create(triplethitsfactoryName, triplethitsfactoryPSet, iC),
      nullptr,
      SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", creatorPSet));

  edm::ParameterSet MixedPSet = pset.getParameter<edm::ParameterSet>("MixedGeneratorSet");
  edm::ParameterSet mixedhitsfactoryPSet = MixedPSet.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string mixedhitsfactoryName = mixedhitsfactoryPSet.getParameter<std::string>("ComponentName");

  theMixedGenerator = std::make_unique<SeedGeneratorFromRegionHits>(
      OrderedHitsGeneratorFactory::get()->create(mixedhitsfactoryName, mixedhitsfactoryPSet, iC),
      nullptr,
      SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", creatorPSet));
}

TSGSmart::~TSGSmart() = default;

void TSGSmart::run(TrajectorySeedCollection &seeds,
                   const edm::Event &ev,
                   const edm::EventSetup &es,
                   const TrackingRegion &region) {
  if (fabs(region.direction().eta()) > theEtaBound) {
    theMixedGenerator->run(seeds, region, ev, es);
  } else {
    theTripletGenerator->run(seeds, region, ev, es);
    if (seeds.empty())
      thePairGenerator->run(seeds, region, ev, es);
  }
}
