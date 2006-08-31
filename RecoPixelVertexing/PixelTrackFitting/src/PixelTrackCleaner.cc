#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

PixelTrackCleaner::PixelTrackCleaner()
{
}


vector<TrackHitsPair> PixelTrackCleaner::cleanTracks(vector<TrackHitsPair> trackHitPairs)
{
  typedef std::vector<const TrackingRecHit *> RecHits;

  LogDebug("PixelTrackCleaner") << "Cleanering tracks" << "\n";
  int size = trackHitPairs.size();
  vector<bool> trackOk;
  for (int i = 0; i < size; i++) trackOk.push_back(true);

  for (int i = 0; i < size; i++)
  {
    const reco::Track *track1 = trackHitPairs.at(i).first;
    RecHits recHits1 = trackHitPairs.at(i).second;

    if (!trackOk.at(i)) continue;

    for (int k = i + 1; k < size; k++)
    {
      if (!trackOk.at(i) || !trackOk.at(k)) continue;

      const reco::Track *track2 = trackHitPairs.at(k).first;
      RecHits recHits2 = trackHitPairs.at(k).second;

      int commonRecHits = 0;
      for (int iRecHit1 = 0; iRecHit1 < (int)recHits1.size(); iRecHit1++)
      {
        for (int iRecHit2 = 0; iRecHit2 < (int)recHits2.size(); iRecHit2++)
        {
          if ((recHits1.at(iRecHit1))->geographicalId() == (recHits2.at(iRecHit2))->geographicalId()) commonRecHits++;
        }
      }
      if (commonRecHits > 1)
      {
        if (track1->pt() > track2->pt()) trackOk.at(k) = false;
        else trackOk.at(i) = false;
      }
    }
  }

  vector<TrackHitsPair> cleanedTracks;

  for (int i = 0; i < size; i++)
  {
    if (trackOk.at(i)) cleanedTracks.push_back(trackHitPairs.at(i));
  }
  return cleanedTracks;
}

