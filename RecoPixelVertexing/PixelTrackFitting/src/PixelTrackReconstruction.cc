#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackReconstruction.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerWrapper.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include <vector>

using namespace pixeltrackfitting;
using namespace ctfseeding;
using edm::ParameterSet;

PixelTrackReconstruction::PixelTrackReconstruction(const ParameterSet& cfg)
  : theConfig(cfg), theFitter(0), theFilter(0), theCleaner(0), theGenerator(0), theRegionProducer(0)
{ }

PixelTrackReconstruction::~PixelTrackReconstruction() 
{
  halt();
}

void PixelTrackReconstruction::halt()
{
  delete theFilter; theFilter=0;
  delete theFitter; theFitter=0;
  delete theCleaner; theCleaner=0;
  delete theGenerator; theGenerator=0;
  delete theRegionProducer; theRegionProducer=0;
}

void PixelTrackReconstruction::init(const edm::EventSetup& es)
{
  ParameterSet regfactoryPSet = theConfig.getParameter<ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer = TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet);

  ParameterSet orderedPSet =
      theConfig.getParameter<ParameterSet>("OrderedHitsFactoryPSet");
  std::string orderedName = orderedPSet.getParameter<std::string>("ComponentName");
  theGenerator = OrderedHitsGeneratorFactory::get()->create( orderedName, orderedPSet);

  ParameterSet fitterPSet = theConfig.getParameter<ParameterSet>("FitterPSet");
  std::string fitterName = fitterPSet.getParameter<std::string>("ComponentName");
  theFitter = PixelFitterFactory::get()->create( fitterName, fitterPSet);

  ParameterSet filterPSet = theConfig.getParameter<ParameterSet>("FilterPSet");
  std::string  filterName = filterPSet.getParameter<std::string>("ComponentName");
  if (filterName != "none") {
    theFilter = theConfig.getParameter<bool>("useFilterWithES") ?
      PixelTrackFilterWithESFactory::get()->create( filterName, filterPSet, es) :
      PixelTrackFilterFactory::get()->create( filterName, filterPSet);
  }

  // seedmerger & its cfg
  edm::ParameterSet mergerPSet = theConfig.getParameter<edm::ParameterSet>( "SeedMergerPSet" );
  std::string seedmergerTTRHBuilderLabel = mergerPSet.getParameter<std::string>( "ttrhBuilderLabel" );
  std::string seedmergerLayerListName = mergerPSet.getParameter<std::string>( "layerListName" );
  bool seedmergerAddTriplets = mergerPSet.getParameter<bool>( "addRemainingTriplets" );
  bool seedmergerMergeTriplets = mergerPSet.getParameter<bool>( "mergeTriplets" );
  theMerger_ = new QuadrupletSeedMerger( es );
  theMerger_->setMergeTriplets( seedmergerMergeTriplets );
  theMerger_->setAddRemainingTriplets( seedmergerAddTriplets );
  theMerger_->setTTRHBuilderLabel( seedmergerTTRHBuilderLabel );
  theMerger_->setLayerListName( seedmergerLayerListName );

  ParameterSet cleanerPSet = theConfig.getParameter<ParameterSet>("CleanerPSet");
  std::string  cleanerName = cleanerPSet.getParameter<std::string>("ComponentName");
  if (cleanerName != "none") theCleaner = PixelTrackCleanerFactory::get()->create( cleanerName, cleanerPSet);

}



void PixelTrackReconstruction::run(TracksWithTTRHs& tracks, edm::Event& ev, const edm::EventSetup& es)
{
  typedef std::vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(ev,es);


  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) {
    const TrackingRegion & region = **ir;

    const OrderedSeedingHits & triplets =  theGenerator->run(region,ev,es);

    ///
    /// the new triplet merger
    ///
    const std::vector<SeedingHitSet> quadruplets = theMerger_->mergeTriplets( triplets, es );

    // producing tracks

    unsigned int nQuad=quadruplets.size();
    tracks.reserve(tracks.size()+nQuad);
    for( std::vector<SeedingHitSet >::const_iterator qIt = quadruplets.begin();
    	 qIt < quadruplets.end(); ++qIt ) {

    //    for (unsigned int iTriplet = 0; iTriplet < nTriplets; ++iTriplet) {
    //      const SeedingHitSet & triplet = triplets[iTriplet];
      const SeedingHitSet& quadruplet = (*qIt);

      std::vector<const TrackingRecHit *> hits;
      unsigned int nHit=quadruplet.size();
      for (unsigned int iHit = 0;  iHit < 4; ++iHit) {
	hits.push_back(quadruplet[iHit]->hit());
      }
    
      // fitting
      reco::Track* track = theFitter->run(es, hits, region);

      // decide if track should be skipped according to filter
      if (theFilter && !(*theFilter)(track, hits) ) {
        delete track;
        continue;
      }

      // add tracks
      tracks.push_back( TrackWithTTRHs( track, quadruplet ) );
    }
    theGenerator->clear();
  }

  // skip ovelrapped tracks
  if (theCleaner) tracks = PixelTrackCleanerWrapper(theCleaner).clean(tracks);

  // clean memory
  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) delete (*ir);
}


