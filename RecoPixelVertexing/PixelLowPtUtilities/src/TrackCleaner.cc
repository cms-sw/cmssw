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
bool TrackCleaner::areSame(const TrackingRecHit * a,
                           const TrackingRecHit * b)
{
  if(a->geographicalId() != b->geographicalId())
    return false;

  if(fabs(a->localPosition().x() - b->localPosition().x()) < 1e-5 &&
     fabs(a->localPosition().y() - b->localPosition().y()) < 1e-5)
    return true;
  else
    return false;
}

/*****************************************************************************/
bool TrackCleaner::isCompatible(const DetId & i1,
                                const DetId & i2)
{
  // different subdet
  if(i1.subdetId() != i2.subdetId()) return true;

  if(i1.subdetId() == int(PixelSubdetector::PixelBarrel))
  { // barrel
    PXBDetId p1(i1);
    PXBDetId p2(i2);

    if(p1.layer() != p2.layer()) return true;

    int dphi = abs(int(p1.ladder() - p2.ladder()));
    static int max[3] = {20, 32, 44};
    if(dphi > max[p1.layer()-1] / 2) dphi = max[p1.layer()-1] - dphi;

    int dz   = abs(int(p1.module() - p2.module()));

    if(dphi == 1 && dz <= 1) return true;
  }
  else
  { // endcap
    PXFDetId p1(i1);
    PXFDetId p2(i2);

    if(p1.side() != p2.side() ||
       p1.disk() != p2.disk()) return true;

    int dphi = abs(int(p1.blade() - p2.blade()));
    static int max = 24;
    if(dphi > max / 2) dphi = max - dphi;

    int dr   = abs(int( ((p1.module()-1) * 2 + (p1.panel()-1)) -
                        ((p2.module()-1) * 2 + (p2.panel()-1)) ));

    if(dphi <= 1 && dr <= 1 && !(dphi == 0 && dr == 0)) return true;
  }

  return false;
}

/*****************************************************************************/
bool TrackCleaner::canBeMerged
  (vector<const TrackingRecHit *> recHitsA,
   vector<const TrackingRecHit *> recHitsB)
{
 bool ok = true;

 for(vector<const TrackingRecHit *>::const_iterator
     recHitA = recHitsA.begin(); recHitA!= recHitsA.end(); recHitA++)
 for(vector<const TrackingRecHit *>::const_iterator
     recHitB = recHitsB.begin(); recHitB!= recHitsB.end(); recHitB++)
   if(!areSame(*recHitA,*recHitB))
     if(!isCompatible((*recHitA)->geographicalId(),
                      (*recHitB)->geographicalId()))
        ok = false;

  return ok;
}

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
  LogTrace("TrackCleaner")
    << "   Track #" << i << " : " << HitInfo::getInfo(tracks[i].second);

  do
  {
  changes = 0;

  RecHitMap recHitMap;

  LogTrace("MinBiasTracking")
    << " [TrackCleaner] fill rechit map";

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

    bool addedNewHit = false;

/*
    do
    {
*/
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
        if(i < *j)
           trackMap[*j]++;
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
          if(canBeMerged(tracks[i].second,tracks[j].second))
          { // no common layer
            // merge tracks, add separate hits of the second to the first one
            for(vector<const TrackingRecHit *>::const_iterator
                recHit = tracks[j].second.begin();
                recHit!= tracks[j].second.end(); recHit++)
            {
              bool ok = true;
              for(vector<const TrackingRecHit *>::const_iterator
                recHitA = tracks[i].second.begin();
                recHitA!= tracks[i].second.end(); recHitA++)
                if(areSame(*recHit,*recHitA)) ok = false;

              if(ok)
              {
                tracks[i].second.push_back(*recHit);
                recHitMap[*recHit].push_back(i);
                addedNewHit = true;
              }
            }

            LogTrace("TrackCleaner") 
              << "   Merge #" << i << " #" << j
              << ", first now has " << tracks[i].second.size();
  
            // Remove second track
            keep[j] = false;
  
           changes++;
          }
          else
          { // there is a common layer, keep smaller impact
            if(fabs(tracks[i].first->d0())
             < fabs(tracks[j].first->d0()))
              keep[j] = false;
            else
              keep[i] = false;

            LogTrace("TrackCleaner")
              << "   Clash #" << i << " #" << j
              << " keep lower d0 " << tracks[i].first->d0()
                            << " " << tracks[j].first->d0()
              << ", keep #" << (keep[i] ? i : ( keep[j] ? j : 9999 ) );
  
            changes++;
          }
        }
        else
        { // note more than 50%, but at least two are shared
          if((*sharing).second > 1)
          {
            if(tracks[i].second.size() != tracks[j].second.size())
            { // keep longer
              if(tracks[i].second.size() > tracks[j].second.size()) 
                keep[j] = false; else keep[i] = false; 
              changes++;

              LogTrace("TrackCleaner")
                << "   Sharing " << (*sharing).second << " remove by size";
            }
            else
            { // keep smaller impact
              if(fabs(tracks[i].first->d0())
               < fabs(tracks[j].first->d0()))
                keep[j] = false; else keep[i] = false; 
              changes++;

              LogTrace("TrackCleaner")
                << "   Sharing " << (*sharing).second << " remove by d0";
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
/*
    }
    while(addedNewHit);
*/
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
  LogTrace("TrackCleaner")
    << "   Track #" << i << " : " << HitInfo::getInfo(cleaned[i].second);

  return cleaned;
}

