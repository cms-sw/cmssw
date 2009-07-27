#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackCleaner.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

using namespace std;
using namespace pixeltrackfitting;

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
TrackCleaner::TrackCleaner
  (const edm::ParameterSet& ps)
{
}

/*****************************************************************************/
TrackCleaner::~TrackCleaner()
{
}

/*****************************************************************************/
int TrackCleaner::getLayer(const DetId & id)
{
  if(id.subdetId() == int(PixelSubdetector::PixelBarrel))
  {
    PXBDetId pid(id);
    return 0 + ((pid.layer() - 1)*2) + ((pid.ladder() - 1)%2);
  } 
  else
  {
    PXFDetId pid(id);
    return 6 + ((pid.disk()  - 1)*2) + ((pid.panel()  - 1)%2);
  } 
}

/*****************************************************************************/
bool TrackCleaner::hasCommonDetUnit
  (vector<const TrackingRecHit *> recHitsA,
   vector<const TrackingRecHit *> recHitsB,
   vector<DetId> detIds)
{
  for(vector<const TrackingRecHit *>::const_iterator
      recHit = recHitsB.begin(); recHit!= recHitsB.end(); recHit++)
    if(find(recHitsA.begin(), recHitsA.end(), *recHit) == recHitsA.end())
    if(find(detIds.begin(),detIds.end(),
            (*recHit)->geographicalId()) != detIds.end())
      return true;

  return false;
}

/*****************************************************************************/
bool TrackCleaner::hasCommonLayer
  (vector<const TrackingRecHit *> recHitsA,
   vector<const TrackingRecHit *> recHitsB,
   vector<int> detLayers)
{
  for(vector<const TrackingRecHit *>::const_iterator
      recHit = recHitsB.begin(); recHit!= recHitsB.end(); recHit++)
    if(find(recHitsA.begin(), recHitsA.end(), *recHit) == recHitsA.end())
    if(find(detLayers.begin(),detLayers.end(),
            getLayer((*recHit)->geographicalId())) != detLayers.end())
      return true;

  return false;
}

/*****************************************************************************/
struct RadiusComparator
{ 
  bool operator() (const TrackingRecHit * h1,
                   const TrackingRecHit * h2)
  { 
    return (h1 < h2);
  };
};

/*****************************************************************************/
TracksWithRecHits TrackCleaner::cleanTracks
  (const TracksWithRecHits & tracks_)
{
  // Local copy
  TracksWithRecHits tracks = tracks_;

  typedef map<const TrackingRecHit*,vector<unsigned int>,HitComparator>
    RecHitMap;

  vector<bool> keep(tracks.size(),true);

  int changes;

  LogTrace("MinBiasTracking")
    << " [TrackCleaner] initial tracks : " << tracks.size();

  for(unsigned int i = 0; i < tracks.size(); i++)
  LogTrace("MinBiasTracking")
    << "   Track #" << i << " : " << HitInfo::getInfo(tracks[i].second);

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

      if(tracks[i].second.size() >=3) 
      { // triplet tracks
        if((*sharing).second > min(int(tracks[i].second.size()),
                                   int(tracks[j].second.size()))/2)
        { // more than min(hits1,hits2)/2 rechits are shared
          if(!hasCommonLayer(tracks[i].second,tracks[j].second,detLayers))
          { 
            // merge tracks, add separate hits of the second to the first one
            for(vector<const TrackingRecHit *>::const_iterator
                recHit = tracks[j].second.begin();
                recHit!= tracks[j].second.end(); recHit++)
              if(find(tracks[i].second.begin(),
                      tracks[i].second.end(),*recHit) == tracks[i].second.end())
                tracks[i].second.push_back(*recHit);

            LogTrace("MinBiasTracking") << "   Merge #" << i << " #" << j;
  
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

            LogTrace("MinBiasTracking") << "   Clash #" << i << " #" << j << " keep lower chi2";
  
            changes++;
          }
        }
        else
        {
          if((*sharing).second > 1)
          {
            if(tracks[i].second.size() != tracks[j].second.size())
            {
              if(tracks[i].second.size() > tracks[j].second.size()) 
                keep[j] = false; else keep[i] = false; 
              changes++;
            LogTrace("MinBiasTracking") << "   Sharing " << (*sharing).second << " remove by size";
            }
            else
            { 
              if(tracks[i].first->chi2() < tracks[j].first->chi2())
                keep[j] = false; else keep[i] = false; 
              changes++;
            LogTrace("MinBiasTracking") << "   Sharing " << (*sharing).second << " remove by chi2";
            } 
          }
        }
      }
      else
      { // pair tracks
        if((*sharing).second > 0)
        {
          // Remove second track
          keep[j] = false;
  
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
    if(keep[i]) 
      cleaned.push_back(tracks[i]);
    else delete tracks_[i].first;

  LogTrace("MinBiasTracking")
    << " [TrackCleaner] cleaned tracks : " << cleaned.size();

  for(unsigned int i = 0; i < cleaned.size(); i++)
  LogTrace("MinBiasTracking")
    << "   Track #" << i << " : " << HitInfo::getInfo(cleaned[i].second);

  return cleaned;
}

