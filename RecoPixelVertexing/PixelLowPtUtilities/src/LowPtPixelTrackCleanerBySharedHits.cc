#include "RecoPixelVertexing/PixelLowPtUtilities/interface/LowPtPixelTrackCleanerBySharedHits.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace reco;
using namespace pixeltrackfitting;

LowPtPixelTrackCleanerBySharedHits::LowPtPixelTrackCleanerBySharedHits( const edm::ParameterSet& cfg)
{}

LowPtPixelTrackCleanerBySharedHits::~LowPtPixelTrackCleanerBySharedHits()
{}

TracksWithRecHits LowPtPixelTrackCleanerBySharedHits::cleanTracks(const
TracksWithRecHits & trackHitPairs_)
{
  typedef std::vector<const TrackingRecHit *> RecHits;
  trackOk.clear();

  // Local copy of trackHitPairs
  TracksWithRecHits trackHitPairs = trackHitPairs_;

  LogDebug("LowPtPixelTrackCleanerBySharedHits") << "Cleaning tracks" << "\n";
  int size = trackHitPairs.size();
  for (int i = 0; i < size; i++) trackOk.push_back(true);

  for (iTrack1 = 0; iTrack1 < size; iTrack1++)
  {
    track1 = trackHitPairs.at(iTrack1).first;
//    RecHits recHits1 = trackHitPairs.at(iTrack1).second;

    if (!trackOk.at(iTrack1)) continue;

    for (iTrack2 = iTrack1 + 1; iTrack2 < size; iTrack2++)
    {
      RecHits recHits1 = trackHitPairs.at(iTrack1).second;
      if (!trackOk.at(iTrack1) || !trackOk.at(iTrack2)) continue;

      track2 = trackHitPairs.at(iTrack2).first;
      RecHits recHits2 = trackHitPairs.at(iTrack2).second;

      vector<int> separateRecHits;

      int commonRecHits = 0;
      for (int iRecHit2 = 0; iRecHit2 < (int)recHits2.size(); iRecHit2++)
      {
        bool match = false;

        for (int iRecHit1 = 0; iRecHit1 < (int)recHits1.size(); iRecHit1++)
          if (recHitsAreEqual(recHits1.at(iRecHit1), recHits2.at(iRecHit2)))
          {
            match = true;
            commonRecHits++;
          }

        if(!match) separateRecHits.push_back(iRecHit2);
      }

      // At least 2 rechits are shared -> add hits, remove second track
      if((int)recHits2.size() - separateRecHits.size() >=2)
      {
        if(fabs(track1->d0() - track2->d0()) < 0.1)
        { // merge
          for(vector<int>::iterator iRecHit2 = separateRecHits.begin();
                                    iRecHit2!= separateRecHits.end();
                                    iRecHit2++)
            trackHitPairs.at(iTrack1).second.push_back(recHits2.at(*iRecHit2));

          cleanTrack();
        }
        else
        { // remove track with higher impact
          if(fabs(track1->d0()) < fabs(track2->d0()))
            trackOk.at(iTrack2) = false;
          else
            trackOk.at(iTrack1) = false;
        }
      }
    }
  }

  vector<TrackWithRecHits> cleanedTracks;

  for (int i = 0; i < size; i++)
  {
    if (trackOk.at(i)) cleanedTracks.push_back(trackHitPairs.at(i));
    else delete trackHitPairs_.at(i).first;
  }
  return cleanedTracks;
}

void LowPtPixelTrackCleanerBySharedHits::cleanTrack()
{
  trackOk.at(iTrack2) = false;
}


bool LowPtPixelTrackCleanerBySharedHits::recHitsAreEqual(const TrackingRecHit *recHit1, const TrackingRecHit *recHit2)
{
  if (recHit1->geographicalId() != recHit2->geographicalId()) return false;
  LocalPoint pos1 = recHit1->localPosition();
  LocalPoint pos2 = recHit2->localPosition();
  return ((pos1.x() == pos2.x()) && (pos1.y() == pos2.y()));
}
