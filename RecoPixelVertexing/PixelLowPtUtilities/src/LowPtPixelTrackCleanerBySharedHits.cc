#include "RecoPixelVertexing/PixelLowPtUtilities/interface/LowPtPixelTrackCleanerBySharedHits.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

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
TracksWithRecHits LowPtPixelTrackCleanerBySharedHits::cleanTracks
  (const TracksWithRecHits & tracks_)
{
  // Local copy
  TracksWithRecHits tracks = tracks_;

  // Fill the rechit map
  typedef map<const TrackingRecHit*,vector<unsigned int>,HitComparator>
    RecHitMap;
  RecHitMap recHitMap;

  vector<bool> keep(tracks.size(),true);

  for(unsigned int i = 0; i < tracks.size(); i++)
  {
    for(vector<const TrackingRecHit *>::const_iterator
        recHit = tracks[i].second.begin();
        recHit!= tracks[i].second.end(); recHit++)
      recHitMap[*recHit].push_back(i);
  }

  cerr << " [TrackCleaner ] initial tracks : " << tracks.size()
                          << " (with " << recHitMap.size() << " hits)" << endl;

  // Look at each track
  typedef map<unsigned int,int,less<unsigned int> > TrackMap; 
  vector<int> ntracks(3,0);

  for(unsigned int i = 0; i < tracks.size(); i++)
  {
    // Skip if 'i' already removed
    if(!keep[i]) continue;

    TrackMap trackMap;

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
    }

    // Check for tracks with shared rechits
    for(TrackMap::iterator sharing = trackMap.begin();
                           sharing!= trackMap.end(); sharing++)
    {
      unsigned int j = (*sharing).first;
      if(!keep[i] || !keep[j]) continue;

      // Old: At least 2 rechits shared
      // if((*sharing).second >= 2)
      // New: More than min(hits1,hits2)/2 rechits are shared

      if((*sharing).second > min(tracks[i].second.size(),
                                 tracks[j].second.size())/2)
      {
        if(fabs(tracks[i].first->d0() - tracks[j].first->d0()) < 0.1)
        { // merge tracks, add separate hits of the second to the first one
          for(vector<const TrackingRecHit *>::const_iterator
              recHit = tracks[j].second.begin();
              recHit!= tracks[j].second.end(); recHit++)
            if(find(tracks[i].second.begin(),
                    tracks[i].second.end(),*recHit) == tracks[i].second.end())
              tracks[i].second.push_back(*recHit);

          // Remove second track
          keep[j] = false;
        }
        else
        { // remove track with higher impact
          if(tracks[i].first->d0() < tracks[j].first->d0())
            keep[j] = false;
          else
            keep[i] = false;
        }
      }

      ntracks[(*sharing).second]++;
    }
  }

  // Final copy
  TracksWithRecHits cleaned;
  
  for(unsigned int i = 0; i < tracks.size(); i++)
    if(keep[i]) cleaned.push_back(tracks[i]);
                      else delete tracks_[i].first;

  cerr << " [TrackCleaner ] cleaned tracks : " << cleaned.size() << endl;


  return cleaned;
}
