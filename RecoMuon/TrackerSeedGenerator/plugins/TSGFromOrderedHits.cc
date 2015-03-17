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
  edm::ParameterSet hitsfactoryPSet =
      theConfig.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");
  OrderedHitsGenerator*  hitsGenerator =
        OrderedHitsGeneratorFactory::get()->create( hitsfactoryName, hitsfactoryPSet, iC);

  edm::ParameterSet creatorPSet;
  creatorPSet.addParameter<std::string>("propagator","PropagatorWithMaterial");
  creatorPSet.addParameter<double>("SeedMomentumForBOFF",5.0);

  std::cout << "[TSGFromOrderedHits::TSGFromOrderedHits] theConfig: " << theConfig << std::endl;

  edm::ParameterSet seedCreatorPSet = theConfig.getParameter<edm::ParameterSet>("SeedCreatorPSet");
  std::cout << "[TSGFromOrderedHits::TSGFromOrderedHits] seedCreatorPSet: " << seedCreatorPSet << std::endl;
  std::string seedCreatorType = seedCreatorPSet.getParameter<std::string>("ComponentName");
  std::cout << " ==> seedCreatorType: " << seedCreatorType << std::endl;

  theGenerator = new SeedGeneratorFromRegionHits(hitsGenerator, 0, 
						 //						 SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", creatorPSet)
						 //						 SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", seedCreatorPSet)
						 SeedCreatorFactory::get()->create(seedCreatorType, seedCreatorPSet)
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
