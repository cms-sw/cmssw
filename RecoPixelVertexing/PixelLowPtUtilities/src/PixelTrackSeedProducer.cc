#include "RecoPixelVertexing/PixelLowPtUtilities/interface/PixelTrackSeedProducer.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/PixelTrackSeedGenerator.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

/*****************************************************************************/
PixelTrackSeedProducer::PixelTrackSeedProducer(const edm::ParameterSet& ps_) :
ps(ps_)
{
  tripletList = ps.getParameter<vector<string> >("tripletList");

  edm::LogInfo("PixelTrackSeedProducer") << " constructor";
  produces<TrajectorySeedCollection>();
}

/*****************************************************************************/
PixelTrackSeedProducer::~PixelTrackSeedProducer()
{
  edm::LogInfo("PixelTrackSeedProducer") << " destructor";
}

/*****************************************************************************/
void PixelTrackSeedProducer::produce
  (edm::Event& ev, const edm::EventSetup& es)
{
  std::cerr << "[Pixel tracks seeds]" << std::endl;

  PixelTrackSeedGenerator thePixelTrackSeedGenerator(es);

  std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection);

  for(vector<string>::const_iterator label = tripletList.begin();
                                     label!= tripletList.end(); label++)
  {
    std::cerr << " [PixelSeeder] " << *label << std::endl;
    edm::Handle<reco::TrackCollection> recCollection;
    ev.getByLabel(*label,recCollection);
    const reco::TrackCollection* recTracks = recCollection.product();
  
    for(reco::TrackCollection::const_iterator track = recTracks->begin();
                                              track!= recTracks->end();
                                              track++)
    {
      TrajectorySeed theSeed = thePixelTrackSeedGenerator.seed(*track,es,ps);
  
      if(theSeed.nHits() >= 2)
        result->push_back(theSeed);
    } 
  }

  std::cerr << " [PixelSeeder] number of seeds : " << result->size() << std::endl;

  // Put result back to the event
  ev.put(result);
}

