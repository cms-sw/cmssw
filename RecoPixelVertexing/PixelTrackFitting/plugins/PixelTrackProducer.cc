#include "PixelTrackProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include <vector>

using namespace pixeltrackfitting;
using edm::ParameterSet;

PixelTrackProducer::PixelTrackProducer(const ParameterSet& cfg)
  : theReconstruction(cfg, consumesCollector())
{
  edm::LogInfo("PixelTrackProducer")<<" construction...";
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}

PixelTrackProducer::~PixelTrackProducer() { }

void PixelTrackProducer::endRun(const edm::Run &run, const edm::EventSetup& es)
{ 
  theReconstruction.halt();
}

void PixelTrackProducer::beginRun(const edm::Run &run, const edm::EventSetup& es)
{
  theReconstruction.init(es);
}

void PixelTrackProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  LogDebug("PixelTrackProducer, produce")<<"event# :"<<ev.id();

  TracksWithTTRHs tracks;
  theReconstruction.run(tracks,ev,es);

  // store tracks
  store(ev, tracks);
}

void PixelTrackProducer::store(edm::Event& ev, const TracksWithTTRHs& tracksWithHits)
{
  std::auto_ptr<reco::TrackCollection> tracks(new reco::TrackCollection());
  std::auto_ptr<TrackingRecHitCollection> recHits(new TrackingRecHitCollection());
  std::auto_ptr<reco::TrackExtraCollection> trackExtras(new reco::TrackExtraCollection());

  int cc = 0, nTracks = tracksWithHits.size();

  for (int i = 0; i < nTracks; i++)
  {
    reco::Track* track =  tracksWithHits.at(i).first;
    const SeedingHitSet& hits = tracksWithHits.at(i).second;

    for (unsigned int k = 0; k < hits.size(); k++)
    {
      TrackingRecHit *hit = hits[k]->hit()->clone();

      track->appendHitPattern(*hit);
      recHits->push_back(hit);
    }
    tracks->push_back(*track);
    delete track;

  }

  LogDebug("TrackProducer") << "put the collection of TrackingRecHit in the event" << "\n";
  edm::OrphanHandle <TrackingRecHitCollection> ohRH = ev.put( recHits );


  for (int k = 0; k < nTracks; k++)
  {
    reco::TrackExtra* theTrackExtra = new reco::TrackExtra();

    //fill the TrackExtra with TrackingRecHitRef
    unsigned int nHits = tracks->at(k).numberOfValidHits();
    for(unsigned int i = 0; i < nHits; ++i) {
      theTrackExtra->add(TrackingRecHitRef(ohRH,cc));
      cc++;
    }

    trackExtras->push_back(*theTrackExtra);
    delete theTrackExtra;
  }

  LogDebug("TrackProducer") << "put the collection of TrackExtra in the event" << "\n";
  edm::OrphanHandle<reco::TrackExtraCollection> ohTE = ev.put(trackExtras);

  for (int k = 0; k < nTracks; k++)
  {
    const reco::TrackExtraRef theTrackExtraRef(ohTE,k);
    (tracks->at(k)).setExtra(theTrackExtraRef);
  }

  ev.put(tracks);

}

void
PixelTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("useFilterWithES",false)->setComment("");
  {
    edm::ParameterSetDescription psd0;
    psd0.add<double>("nSigmaInvPtTolerance",0.0)->setComment("");
    psd0.add<double>("chi2",1000.0)->setComment("");
    psd0.add<std::string>("ComponentName","PixelTrackFilterByKinematics")->setComment("");
    psd0.add<double>("nSigmaTipMaxTolerance",0.0)->setComment("");
    psd0.add<double>("ptMin",0.1)->setComment("");
    psd0.add<double>("tipMax",1.0)->setComment("");

    desc.add<edm::ParameterSetDescription>("FilterPSet",psd0)->setComment("");
  }
  desc.add<std::string>("passLabel","pixelTracks")->setComment("");
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("ComponentName","PixelFitterByHelixProjections")->setComment("");
    psd0.add<std::string>("TTRHBuilder","PixelTTRHBuilderWithoutAngle")->setComment("");

    desc.add<edm::ParameterSetDescription>("FitterPSet",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("ComponentName","GlobalRegionProducerFromBeamSpot")->setComment("");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<bool>("precise",true)->setComment("");
      psd1.add<double>("originRadius",0.2)->setComment("");
      psd1.add<double>("nSigmaZ",4.0)->setComment("");
      psd1.add<edm::InputTag>("beamSpot",edm::InputTag("offlineBeamSpot"))->setComment("");
      psd1.add<double>("ptMin",0.9)->setComment("");

      psd0.add<edm::ParameterSetDescription>("RegionPSet",psd1)->setComment("");
    }
    desc.add<edm::ParameterSetDescription>("RegionFactoryPSet",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("ComponentName","PixelTrackCleanerBySharedHits")->setComment("");
    desc.add<edm::ParameterSetDescription>("CleanerPSet",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("ComponentName","StandardHitTripletGenerator")->setComment("");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<bool>("useBending",true)->setComment("");
      psd1.add<bool>("useFixedPreFiltering",false)->setComment("");
      psd1.add<unsigned int>("maxElement",100000)->setComment("");
      {
        edm::ParameterSetDescription psd2;
        psd2.add<edm::InputTag>("clusterShapeCacheSrc",edm::InputTag("siPixelClusterShapeCache"))->setComment("");
        psd2.add<std::string>("ComponentName","LowPtClusterShapeSeedComparitor")->setComment("");
        psd1.add<edm::ParameterSetDescription>("SeedComparitorPSet",psd2)->setComment("");
      }
      psd1.add<double>("extraHitRPhitolerance",0.032)->setComment("");
      psd1.add<bool>("useMultScattering",true)->setComment("");
      psd1.add<double>("phiPreFiltering",0.3)->setComment("");
      psd1.add<double>("extraHitRZtolerance",0.037)->setComment("");
      psd1.add<std::string>("ComponentName","PixelTripletHLTGenerator")->setComment("");
      psd0.add<edm::ParameterSetDescription>("GeneratorPSet",psd1)->setComment("");
    }
    psd0.add<edm::InputTag>("SeedingLayers",edm::InputTag("PixelLayerTriplets"))->setComment("");
    desc.add<edm::ParameterSetDescription>("OrderedHitsFactoryPSet",psd0)->setComment("");
  }

  descriptions.add("pixelTracks",desc);
  descriptions.setComment("");
}
