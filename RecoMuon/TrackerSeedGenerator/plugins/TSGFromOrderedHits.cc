#include "TSGFromOrderedHits.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/RunID.h"


TSGFromOrderedHits::TSGFromOrderedHits(const edm::ParameterSet &pset)
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
  theGenerator = new SeedGeneratorFromRegionHits( hitsGenerator, theConfig);
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
