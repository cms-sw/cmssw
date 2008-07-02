#include "RecoPixelVertexing/PixelLowPtUtilities/interface/SeedProducer.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/SeedGenerator.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

/*****************************************************************************/
SeedProducer::SeedProducer(const edm::ParameterSet& ps_) :
ps(ps_)
{
  tripletList = ps.getParameter<vector<string> >("tripletList");

  edm::LogInfo("SeedProducer") << " constructor";
  produces<TrajectorySeedCollection>();
}

/*****************************************************************************/
SeedProducer::~SeedProducer()
{
  edm::LogInfo("SeedProducer") << " destructor";
}

/*****************************************************************************/
void SeedProducer::produce
  (edm::Event& ev, const edm::EventSetup& es)
{
  SeedGenerator theSeedGenerator(es);

  std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection);

  for(vector<string>::const_iterator label = tripletList.begin();
                                     label!= tripletList.end(); label++)
  {
    LogTrace("MinBiasTracking")
      << " [PixelSeeder] " << *label;

    edm::Handle<reco::TrackCollection> recCollection;
    ev.getByLabel(*label,recCollection);
    const reco::TrackCollection* recTracks = recCollection.product();
  
    for(reco::TrackCollection::const_iterator track = recTracks->begin();
                                              track!= recTracks->end();
                                              track++)
    {
      LogTrace("MinBiasTracking")
        << "  [PixelSeeder] track #" << track - recTracks->begin();

      TrajectorySeed theSeed = theSeedGenerator.seed(*track,es,ps);
  
      if(theSeed.nHits() >= 2)
        result->push_back(theSeed);
    } 
  }

  LogTrace("MinBiasTracking")
    << " [PixelSeeder] number of seeds : " << result->size();

  // Put result back to the event
  ev.put(result);
}

