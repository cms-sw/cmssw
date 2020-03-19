#include "TSGFromOrderedHits.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/RunID.h"

TSGFromOrderedHits::TSGFromOrderedHits(const edm::ParameterSet &pset, edm::ConsumesCollector &iC) : theLastRun(0) {
  edm::ParameterSet hitsfactoryPSet = pset.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");

  edm::ParameterSet seedCreatorPSet = pset.getParameter<edm::ParameterSet>("SeedCreatorPSet");
  std::string seedCreatorType = seedCreatorPSet.getParameter<std::string>("ComponentName");

  theGenerator = std::make_unique<SeedGeneratorFromRegionHits>(
      OrderedHitsGeneratorFactory::get()->create(hitsfactoryName, hitsfactoryPSet, iC),
      nullptr,
      SeedCreatorFactory::get()->create(seedCreatorType, seedCreatorPSet));
}

TSGFromOrderedHits::~TSGFromOrderedHits() = default;

void TSGFromOrderedHits::run(TrajectorySeedCollection &seeds,
                             const edm::Event &ev,
                             const edm::EventSetup &es,
                             const TrackingRegion &region) {
  theGenerator->run(seeds, region, ev, es);
}
