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

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerWrapper.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "RecoPixelVertexing/PixelTriplets/interface/QuadrupletSeedMerger.h"

#include <vector>

using namespace pixeltrackfitting;
using namespace ctfseeding;
using edm::ParameterSet;

PixelTrackReconstruction::PixelTrackReconstruction(const ParameterSet& cfg,
	   edm::ConsumesCollector && iC)
  : theFitterToken(iC.consumes<PixelFitter>(cfg.getParameter<edm::InputTag>("Fitter"))),
    theCleanerName(cfg.getParameter<std::string>("Cleaner"))
{
  if ( cfg.exists("SeedMergerPSet") ) {
    edm::ParameterSet mergerPSet = cfg.getParameter<edm::ParameterSet>( "SeedMergerPSet" );
    std::string seedmergerTTRHBuilderLabel = mergerPSet.getParameter<std::string>( "ttrhBuilderLabel" );
    edm::ParameterSet seedmergerLayerList = mergerPSet.getParameter<edm::ParameterSet>( "layerList" );
    bool seedmergerAddTriplets = mergerPSet.getParameter<bool>( "addRemainingTriplets" );
    bool seedmergerMergeTriplets = mergerPSet.getParameter<bool>( "mergeTriplets" );
    theMerger_.reset(new QuadrupletSeedMerger(seedmergerLayerList, iC));
    theMerger_->setMergeTriplets( seedmergerMergeTriplets );
    theMerger_->setAddRemainingTriplets( seedmergerAddTriplets );
    theMerger_->setTTRHBuilderLabel( seedmergerTTRHBuilderLabel );
  }

  edm::InputTag filterTag = cfg.getParameter<edm::InputTag>("Filter");
  if(filterTag.label() != "") {
    theFilterToken = iC.consumes<PixelTrackFilter>(filterTag);
  }

  ParameterSet orderedPSet =
      cfg.getParameter<ParameterSet>("OrderedHitsFactoryPSet");
  std::string orderedName = orderedPSet.getParameter<std::string>("ComponentName");
  theGenerator.reset(OrderedHitsGeneratorFactory::get()->create( orderedName, orderedPSet, iC));

  ParameterSet regfactoryPSet = cfg.getParameter<ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer.reset(TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet, std::move(iC)));
}
  
PixelTrackReconstruction::~PixelTrackReconstruction() 
{
}

void PixelTrackReconstruction::init(const edm::EventSetup& es)
{
  if (theMerger_) {
    theMerger_->update( es );
  }
}

void PixelTrackReconstruction::run(TracksWithTTRHs& tracks, edm::Event& ev, const edm::EventSetup& es)
{
  typedef std::vector<std::unique_ptr<TrackingRegion> > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(ev,es);

  edm::Handle<PixelFitter> hfitter;
  ev.getByToken(theFitterToken, hfitter);
  const auto& fitter = *hfitter;

  const PixelTrackFilter *filter = nullptr;
  if(!theFilterToken.isUninitialized()) {
    edm::Handle<PixelTrackFilter> hfilter;
    ev.getByToken(theFilterToken, hfilter);
    filter = hfilter.product();
  }
  
  std::vector<const TrackingRecHit *> hits;hits.reserve(4); 
  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) {
    const TrackingRegion & region = **ir;

    const OrderedSeedingHits & triplets =  theGenerator->run(region,ev,es);
    const OrderedSeedingHits &tuplets= (theMerger_==0)? triplets : theMerger_->mergeTriplets( triplets, es );

    unsigned int nTuplets = tuplets.size();
    tracks.reserve(tracks.size()+nTuplets);
    // producing tracks
    for (unsigned int iTuplet = 0; iTuplet < nTuplets; ++iTuplet) {
      const SeedingHitSet & tuplet = tuplets[iTuplet];

      /// FIXME at some point we need to migrate the fitter...
      auto nHits = tuplet.size(); hits.resize(nHits);
      for (unsigned int iHit = 0; iHit < nHits; ++iHit) hits[iHit] = tuplet[iHit];
   
      // fitting
      std::unique_ptr<reco::Track> track = fitter.run(hits, region);
      if (!track) continue;

      if (filter) {
	if (!(*filter)(track.get(), hits)) {
	  continue;
	}
      }

      // add tracks
      tracks.emplace_back(track.release(), tuplet);
    }
    theGenerator->clear();
  }

  // skip ovelrapped tracks
  if(!theCleanerName.empty()) {
    edm::ESHandle<PixelTrackCleaner> hcleaner;
    es.get<PixelTrackCleaner::Record>().get(theCleanerName, hcleaner);
    const auto& cleaner = *hcleaner;
    if(cleaner.fast())
      cleaner.cleanTracks(tracks);
    else
      tracks = PixelTrackCleanerWrapper(&cleaner).clean(tracks);
  }
}
