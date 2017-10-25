#include "RecoMuon/TrackerSeedGenerator/plugins/TSGSmart.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"


TSGSmart::TSGSmart(const edm::ParameterSet &pset,edm::ConsumesCollector& iC)
  : theConfig(pset), thePairGenerator(nullptr), theTripletGenerator(nullptr), theMixedGenerator(nullptr)
{

  theEtaBound = theConfig.getParameter<double>("EtaBound");

  // FIXME??
  edm::ParameterSet creatorPSet;
  creatorPSet.addParameter<std::string>("propagator","PropagatorWithMaterial");

  edm::ParameterSet PairPSet = theConfig.getParameter<edm::ParameterSet>("PixelPairGeneratorSet"); 
  edm::ParameterSet pairhitsfactoryPSet =
    PairPSet.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string pairhitsfactoryName = pairhitsfactoryPSet.getParameter<std::string>("ComponentName");
  OrderedHitsGenerator*  pairhitsGenerator =
    OrderedHitsGeneratorFactory::get()->create( pairhitsfactoryName, pairhitsfactoryPSet, iC);


  thePairGenerator = new SeedGeneratorFromRegionHits( pairhitsGenerator, nullptr, 
						 SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", creatorPSet)
						 );

  edm::ParameterSet TripletPSet = theConfig.getParameter<edm::ParameterSet>("PixelTripletGeneratorSet"); 
  edm::ParameterSet triplethitsfactoryPSet =
    TripletPSet.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string triplethitsfactoryName = triplethitsfactoryPSet.getParameter<std::string>("ComponentName");
  OrderedHitsGenerator*  triplethitsGenerator =
    OrderedHitsGeneratorFactory::get()->create( triplethitsfactoryName, triplethitsfactoryPSet, iC);
  theTripletGenerator = new SeedGeneratorFromRegionHits( triplethitsGenerator, nullptr, 
						 SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", creatorPSet)
						 );

  edm::ParameterSet MixedPSet = theConfig.getParameter<edm::ParameterSet>("MixedGeneratorSet"); 
  edm::ParameterSet mixedhitsfactoryPSet =
    MixedPSet.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string mixedhitsfactoryName = mixedhitsfactoryPSet.getParameter<std::string>("ComponentName");
  OrderedHitsGenerator*  mixedhitsGenerator =
    OrderedHitsGeneratorFactory::get()->create( mixedhitsfactoryName, mixedhitsfactoryPSet, iC);
  theMixedGenerator = new SeedGeneratorFromRegionHits( mixedhitsGenerator, nullptr, 
						 SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", creatorPSet)
						 );
  
 
}

TSGSmart::~TSGSmart()
{
  delete thePairGenerator; 
  delete theTripletGenerator; 
  delete theMixedGenerator; 
}

void TSGSmart::run(TrajectorySeedCollection &seeds, 
      const edm::Event &ev, const edm::EventSetup &es, const TrackingRegion& region)
{
  if( fabs(region.direction().eta()) >  theEtaBound ) {
    theMixedGenerator->run(seeds, region, ev, es);
  } else {
    theTripletGenerator->run(seeds, region, ev, es);
    if(seeds.empty()) thePairGenerator->run(seeds, region, ev, es);
  }
}
