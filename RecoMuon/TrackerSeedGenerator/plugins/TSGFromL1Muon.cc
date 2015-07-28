#include "TSGFromL1Muon.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"
#include "RecoMuon/TrackerSeedGenerator/interface/L1MuonPixelTrackFitter.h"
#include "RecoMuon/TrackerSeedGenerator/interface/L1MuonSeedsMerger.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoMuon/TrackerSeedGenerator/interface/L1MuonRegionProducer.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedFromProtoTrack.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include <vector>

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

using namespace reco;
using namespace ctfseeding;
using namespace l1extra;

template <class T> T sqr( T t) {return t*t;}


TSGFromL1Muon::TSGFromL1Muon(const edm::ParameterSet& cfg)
  : theConfig(cfg),theRegionProducer(0),theHitGenerator(0),theFitter(0),theMerger(0)
{
  produces<L3MuonTrajectorySeedCollection>();
  theSourceTag = cfg.getParameter<edm::InputTag>("L1MuonLabel");

  edm::ConsumesCollector iC = consumesCollector();
  edm::ParameterSet filterPSet = theConfig.getParameter<edm::ParameterSet>("FilterPSet");
  std::string  filterName = filterPSet.getParameter<std::string>("ComponentName");
  theFilter.reset(PixelTrackFilterFactory::get()->create( filterName, filterPSet, iC));

  edm::ParameterSet hitsfactoryPSet =
      theConfig.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");
  theHitGenerator = OrderedHitsGeneratorFactory::get()->create( hitsfactoryName, hitsfactoryPSet, iC);

  theSourceToken=iC.consumes<L1MuonParticleCollection>(theSourceTag);
}

TSGFromL1Muon::~TSGFromL1Muon()
{
  delete theMerger;
  delete theFitter;
  delete theHitGenerator;
  delete theRegionProducer;
}

void TSGFromL1Muon::beginRun(const edm::Run & run, const edm::EventSetup&es)
{
  edm::ParameterSet regfactoryPSet = theConfig.getParameter<edm::ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  TrackingRegionProducer * p =
    TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet, consumesCollector());
  theRegionProducer = dynamic_cast<L1MuonRegionProducer* >(p);

  edm::ParameterSet fitterPSet = theConfig.getParameter<edm::ParameterSet>("FitterPSet");
  std::string fitterName = fitterPSet.getParameter<std::string>("ComponentName");
  PixelFitter * f = PixelFitterFactory::get()->create( fitterName, fitterPSet);
  theFitter = dynamic_cast<L1MuonPixelTrackFitter* >(f);

  edm::ParameterSet cleanerPSet = theConfig.getParameter<edm::ParameterSet>("CleanerPSet");
  std::string  cleanerName = cleanerPSet.getParameter<std::string>("ComponentName");
//  theMerger = PixelTrackCleanerFactory::get()->create( cleanerName, cleanerPSet);
  theMerger = new L1MuonSeedsMerger(cleanerPSet);
}


void TSGFromL1Muon::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<L3MuonTrajectorySeedCollection> result(new L3MuonTrajectorySeedCollection());

  edm::Handle<L1MuonParticleCollection> l1muon;
  ev.getByToken(theSourceToken, l1muon);

  LogDebug("TSGFromL1Muon")<<l1muon->size()<<" l1 muons to seed from.";

  L1MuonParticleCollection::const_iterator muItr = l1muon->begin(); 
  L1MuonParticleCollection::const_iterator muEnd = l1muon->end(); 
  for  ( size_t iL1 = 0;  muItr < muEnd; ++muItr, ++iL1) {
       
    if (muItr->gmtMuonCand().empty()) continue;

    const L1MuGMTCand & muon = muItr->gmtMuonCand();
    l1extra::L1MuonParticleRef l1Ref(l1muon, iL1);

    theRegionProducer->setL1Constraint(muon);
    theFitter->setL1Constraint(muon);

    typedef std::vector<std::unique_ptr<TrackingRegion> > Regions;
    Regions regions = theRegionProducer->regions(ev,es);
    for (Regions::const_iterator ir=regions.begin(); ir != regions.end(); ++ir) {

      L1MuonSeedsMerger::TracksAndHits tracks;
      const TrackingRegion & region = **ir;
      const OrderedSeedingHits & candidates = theHitGenerator->run(region,ev,es);

      unsigned int nSets = candidates.size();
      for (unsigned int ic= 0; ic <nSets; ic++) {

        const SeedingHitSet & hits =  candidates[ic]; 
        std::vector<const TrackingRecHit *> trh;
        for (unsigned int i= 0, nHits = hits.size(); i< nHits; ++i) trh.push_back( hits[i]->hit() );

        theFitter->setPxConstraint(hits);
        reco::Track* track = theFitter->run(es, trh, region);
        if (!track) continue;

        if (!(*theFilter)(track) ) { delete track; continue; }
        tracks.push_back(L1MuonSeedsMerger::TrackAndHits(track, hits));
      }
  
      if(theMerger) theMerger->resolve(tracks);
      for (L1MuonSeedsMerger::TracksAndHits::const_iterator it = tracks.begin();
        it != tracks.end(); ++it) {

        SeedFromProtoTrack seed( *(it->first), it->second, es);
        if (seed.isValid()) (*result).push_back(L3MuonTrajectorySeed(seed.trajectorySeed(),l1Ref));

//      GlobalError vtxerr( sqr(region->originRBound()), 0, sqr(region->originRBound()),
//                                               0, 0, sqr(region->originZBound()));
//      SeedFromConsecutiveHits seed( candidates[ic],region->origin(), vtxerr, es);
//      if (seed.isValid()) (*result).push_back( seed.TrajSeed() );
        delete it->first;
      }
    }
  }

  LogDebug("TSGFromL1Muon")<<result->size()<<" seeds to the event.";
  ev.put(result);
}

