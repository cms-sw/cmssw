#include "RecoPixelVertexing/PixelLowPtUtilities/interface/LowPtPixelTrackCleanerBySharedHits.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

/*****************************************************************************/
class HitComparator
{
  public:
    bool operator() (const TrackingRecHit* a, const TrackingRecHit* b) const
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
LowPtPixelTrackCleanerBySharedHits::LowPtPixelTrackCleanerBySharedHits
  (const edm::ParameterSet& ps)
{
}

/*****************************************************************************/
LowPtPixelTrackCleanerBySharedHits::~LowPtPixelTrackCleanerBySharedHits()
{
}

/*****************************************************************************/
int LowPtPixelTrackCleanerBySharedHits::getLayer(const DetId & id)
{
  if(id.subdetId() == PixelSubdetector::PixelBarrel)
  {
    PXBDetId pid(id);
    return 0 + (pid.layer() - 1)<<1 + (pid.ladder() - 1)%2;
  } 
  else
  {
    PXFDetId pid(id);
    return 6 + (pid.disk()  - 1)<<1 + (pid.panel()  - 1)%2;
  } 
}

/*****************************************************************************/
TracksWithRecHits LowPtPixelTrackCleanerBySharedHits::cleanTracks
  (const TracksWithRecHits & tracks_)
{
  // Local copy
  TracksWithRecHits tracks = tracks_;

  typedef map<const TrackingRecHit*,vector<unsigned int>,HitComparator>
    RecHitMap;

  vector<bool> keep(tracks.size(),true);

  int changes;

  cerr << " [TrackCleaner ] initial tracks : " << tracks.size() << endl;

  do
  {
  changes = 0;

  RecHitMap recHitMap;

  // Fill the rechit map
  for(unsigned int i = 0; i < tracks.size(); i++)
  if(keep[i])
  {
    for(vector<const TrackingRecHit *>::const_iterator
        recHit = tracks[i].second.begin();
        recHit!= tracks[i].second.end(); recHit++)
      recHitMap[*recHit].push_back(i);
  }

/*
  cerr << " [TrackCleaner ] initial tracks : " << tracks.size()
                          << " (with " << recHitMap.size() << " hits)" << endl;
*/

  // Look at each track
  typedef map<unsigned int,int,less<unsigned int> > TrackMap; 

  for(unsigned int i = 0; i < tracks.size(); i++)
  {
    // Skip if 'i' already removed
    if(!keep[i]) continue;

    TrackMap trackMap;
    vector<DetId> detIds;
    vector<int> detLayers;

    // Go trough all rechits of this track
    for(vector<const TrackingRecHit *>::const_iterator
        recHit = tracks[i].second.begin();
        recHit!= tracks[i].second.end(); recHit++)
    {
      // Get tracks sharing this rechit
      vector<unsigned int> sharing = recHitMap[*recHit];

      for(vector<unsigned int>::iterator j = sharing.begin();
                                         j!= sharing.end(); j++)
        if(i < *j) trackMap[*j]++;

      // Fill detLayers vector
      detIds.push_back((*recHit)->geographicalId());
      detLayers.push_back(getLayer((*recHit)->geographicalId()));
    }

    // Check for tracks with shared rechits
    for(TrackMap::iterator sharing = trackMap.begin();
                           sharing!= trackMap.end(); sharing++)
    {
      unsigned int j = (*sharing).first;
      if(!keep[i] || !keep[j]) continue;

      // More than min(hits1,hits2)/2 rechits are shared
      if((*sharing).second > min(tracks[i].second.size(),
                                 tracks[j].second.size())/2)
      {
        bool hasCommonDetUnit = false;

        for(vector<const TrackingRecHit *>::const_iterator
              recHit = tracks[j].second.begin();
              recHit!= tracks[j].second.end(); recHit++)
           if(find(tracks[i].second.begin(), tracks[i].second.end(),*recHit)
                                          == tracks[i].second.end())
           if(find(detIds.begin(),detIds.end(),(*recHit)->geographicalId()) 
                               != detIds.end())
             hasCommonDetUnit = true;

        bool hasCommonLayer = false;

        for(vector<const TrackingRecHit *>::const_iterator
              recHit = tracks[j].second.begin();
              recHit!= tracks[j].second.end(); recHit++)
           if(find(tracks[i].second.begin(), tracks[i].second.end(),*recHit)
                                          == tracks[i].second.end())
           if(find(detLayers.begin(),detLayers.end(),
                  getLayer((*recHit)->geographicalId()))
                               != detLayers.end())
             hasCommonLayer = true;

        if(hasCommonLayer == false)
        { 
          // merge tracks, add separate hits of the second to the first one
          for(vector<const TrackingRecHit *>::const_iterator
              recHit = tracks[j].second.begin();
              recHit!= tracks[j].second.end(); recHit++)
            if(find(tracks[i].second.begin(),
                    tracks[i].second.end(),*recHit) == tracks[i].second.end())
              tracks[i].second.push_back(*recHit);

          // Remove second track
          keep[j] = false;

         changes++;
        }
        else
        { // remove track with higher impact / chi2
          if(tracks[i].first->chi2() < tracks[j].first->chi2())
            keep[j] = false;
          else
            keep[i] = false;

          changes++;
        }
      }
    }
  }
  }
  while(changes > 0);

  // Final copy
  TracksWithRecHits cleaned;
  
  for(unsigned int i = 0; i < tracks.size(); i++)
    if(keep[i]) cleaned.push_back(tracks[i]);
                      else delete tracks_[i].first;

  cerr << " [TrackCleaner ] cleaned tracks : " << cleaned.size() << endl;

  return cleaned;
}
