#include "TSGFromOrderedHits.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/RunID.h"


TSGFromOrderedHits::TSGFromOrderedHits(const edm::ParameterSet &pset,edm::ConsumesCollector & iC)
  : theLastRun(0), theConfig(pset), theGenerator(0)
{
	init();
}
void TSGFromOrderedHits::init()
{
  edm::ParameterSet hitsfactoryPSet =
      theConfig.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");
  OrderedHitsGenerator*  hitsGenerator =
        OrderedHitsGeneratorFactory::get()->create( hitsfactoryName, hitsfactoryPSet);

  if (theGenerator) delete theGenerator;
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
  edm::RunNumber_t thisRun = ev.run();
  if (thisRun != theLastRun) { theLastRun = thisRun; init(); }
  theGenerator->run( seeds, region, ev, es);
}
