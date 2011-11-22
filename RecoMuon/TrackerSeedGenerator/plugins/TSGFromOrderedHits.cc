#include "TSGFromOrderedHits.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"

TSGFromOrderedHits::TSGFromOrderedHits(const edm::ParameterSet &pset)
  : theConfig(pset), theGenerator(0)
{
 
  edm::ParameterSet hitsfactoryPSet =
      theConfig.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");
  OrderedHitsGenerator*  hitsGenerator =
        OrderedHitsGeneratorFactory::get()->create( hitsfactoryName, hitsfactoryPSet);

  // FIXME??
  edm::ParameterSet creatorPSet;
  creatorPSet.addParameter<std::string>("propagator","PropagatorWithMaterial");
  theGenerator = new SeedGeneratorFromRegionHits(hitsGenerator, 0, 
						 SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", creatorPSet)
						 );


}

TSGFromOrderedHits::~TSGFromOrderedHits()
{
  delete theGenerator; 
}

void TSGFromOrderedHits::run(TrajectorySeedCollection &seeds, 
      const edm::Event &ev, const edm::EventSetup &es, const TrackingRegion& region)
{
  theGenerator->run( seeds, region, ev, es);
}
