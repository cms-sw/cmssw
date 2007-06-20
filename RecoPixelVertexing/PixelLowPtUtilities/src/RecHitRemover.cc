#include "RecoPixelVertexing/PixelLowPtUtilities/interface/RecHitRemover.h"

#include "DataFormats/TrackReco/interface/Track.h"

/*****************************************************************************/
class HitComparator
{
  public:
    bool operator() (const SiPixelRecHit* a, const SiPixelRecHit* b) const
    {
      if(a->geographicalId() < b->geographicalId()) return true;
      if(b->geographicalId() < a->geographicalId()) return false;

      if(a->localPosition().x() < b->localPosition().x() - 1e-5) return true;
      if(b->localPosition().x() < a->localPosition().x() - 1e-5) return false;

      if(a->localPosition().y() < b->localPosition().y() - 1e-5) return true;
      if(b->localPosition().y() < a->localPosition().y() - 1e-5) return false;

      return false;
    }
};

/*****************************************************************************/
RecHitRemover::RecHitRemover(const edm::ParameterSet& ps)
{
  hitCollectionLabel = ps.getParameter<edm::InputTag>("HitCollectionLabel");
  removeHitsList     = ps.getParameter<vector<string> >("removeHitsList");
}

/*****************************************************************************/
RecHitRemover::~RecHitRemover()
{
}

/*****************************************************************************/
SiPixelRecHitCollection RecHitRemover::getFreeHits(const edm::Event& ev)
{
  // Sets
  set<const SiPixelRecHit*, HitComparator> allHits,usedHits,freeHits;

  // Get hits
  edm::Handle<SiPixelRecHitCollection> pixelCollection;
  ev.getByLabel(hitCollectionLabel,    pixelCollection);
  const SiPixelRecHitCollection* recHits = pixelCollection.product();

  for(SiPixelRecHitCollection::const_iterator recHit = recHits->begin();
                                              recHit!= recHits->end();
                                              recHit++)
  {
    const SiPixelRecHit* pixelHit = &(*recHit);  

    allHits.insert(pixelHit);
  }

  // Get tracks
  edm::Handle<reco::TrackCollection> recCollection;

  for(vector<string>::const_iterator label = removeHitsList.begin();
                                     label!= removeHitsList.end(); label++)
  {
    ev.getByLabel(*label,  recCollection);
    const reco::TrackCollection* recTracks = recCollection.product();
  
    for(reco::TrackCollection::const_iterator recTrack = recTracks->begin();
                                              recTrack!= recTracks->end();
                                              recTrack++)
      for(trackingRecHit_iterator recHit = recTrack->recHitsBegin();
                                  recHit!= recTrack->recHitsEnd();
                                  recHit++)
        if((*recHit)->isValid())
        {
          const SiPixelRecHit* pixelHit =
            dynamic_cast<const SiPixelRecHit *>(&(**recHit));
  
          usedHits.insert(pixelHit);
        }
  }

  // Difference: free = all - used
  set_difference(allHits.begin(), allHits.end(),
                usedHits.begin(),usedHits.end(),
               inserter(freeHits,freeHits.begin()), HitComparator());

  SiPixelRecHitCollection output;

  DetId detId(0);
  edm::OwnVector<SiPixelRecHit> recHitsOnDetUnit;

  for(set<const SiPixelRecHit*, HitComparator>::iterator
      pixelHit = freeHits.begin(); pixelHit!= freeHits.end(); pixelHit++)
  {
    if((*pixelHit)->geographicalId() != detId)
    {
      if(recHitsOnDetUnit.size() > 0)
      { 
        output.put(detId, recHitsOnDetUnit.begin(), recHitsOnDetUnit.end());
        recHitsOnDetUnit.clear();
      }

      detId = (*pixelHit)->geographicalId();
    }

    SiPixelRecHit* recHit = const_cast<SiPixelRecHit*> (*pixelHit);
    recHitsOnDetUnit.push_back(recHit->clone());
  }

  if(recHitsOnDetUnit.size() > 0)
    output.put(detId, recHitsOnDetUnit.begin(), recHitsOnDetUnit.end());

  cerr << " [RecHitRemover] all/used/free hits:"
            << " " <<  allHits.size()
            << "/" << usedHits.size()
            << "/" << output.size()
            << endl;

  return output;
}
