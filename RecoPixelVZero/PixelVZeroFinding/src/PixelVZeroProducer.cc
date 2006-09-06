#include "RecoPixelVZero/PixelVZeroFinding/interface/PixelVZeroProducer.h"
#include "RecoPixelVZero/PixelVZeroFinding/interface/PixelVZeroFinder.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*****************************************************************************/
PixelVZeroProducer::PixelVZeroProducer(const edm::ParameterSet& pset)
  : pset_(pset)
{
  edm::LogInfo("PixelVZeroProducer") << " constructor";
  produces<reco::VZeroCollection>();

  // Get track level cuts
  minImpactPositiveDaughter =
    pset.getParameter<double>("minImpactPositiveDaughter");
  minImpactNegativeDaughter =
    pset.getParameter<double>("minImpactNegativeDaughter");
}

/*****************************************************************************/
PixelVZeroProducer::~PixelVZeroProducer()
{
  edm::LogInfo("PixelVZeroProducer") << " destructor";
}

/*****************************************************************************/
void PixelVZeroProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  LogDebug("PixelVZeroProducer, produce")<<"event# :"<<ev.id();

  edm::Handle<reco::TrackCollection> trackCollection;
  ev.getByLabel("pixelTracks",trackCollection);

  // Get tracks
  const reco::TrackCollection tracks = *(trackCollection.product());

  // Find vzeros
  PixelVZeroFinder theFinder(es,pset_);

  // Selection based on track impact parameter
  reco::TrackRefVector positives;
  reco::TrackRefVector negatives;

  for(unsigned int i=0; i<tracks.size(); i++)
  {
    if(tracks[i].charge() > 0 &&
       fabs(tracks[i].d0()) > minImpactPositiveDaughter)
      positives.push_back(reco::TrackRef(trackCollection, i));

    if(tracks[i].charge() < 0 &&
       fabs(tracks[i].d0()) > minImpactNegativeDaughter)
      negatives.push_back(reco::TrackRef(trackCollection, i));
  }

  std::auto_ptr<reco::VZeroCollection> result(new reco::VZeroCollection);

  // Check all combination of positives and negatives
  if(positives.size() > 0 && negatives.size() > 0)
    for(reco::track_iterator ipos = positives.begin();
                             ipos!= positives.end(); ipos++)
    for(reco::track_iterator ineg = negatives.begin();
                             ineg!= negatives.end(); ineg++)
    {
      reco::VZeroData data;

      if(theFinder.checkTrackPair(**ipos,**ineg, data) == true)
      {
        // Create vertex (creation point)
        reco::Vertex vertex(reco::Vertex::Point(data.crossingPoint.x(),
                                                data.crossingPoint.y(),
                                                data.crossingPoint.z()),
                            reco::Vertex::Error(), 0.,0.,0);

        // Add references to daughters
        vertex.add(*ipos);
        vertex.add(*ineg);

        // Store vzero
        result->push_back(reco::VZero(vertex,data));
      }
    }

  // Put result back to the event
  ev.put(result);
}

DEFINE_FWK_MODULE(PixelVZeroProducer);
