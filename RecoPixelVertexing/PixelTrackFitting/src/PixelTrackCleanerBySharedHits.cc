#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerBySharedHits.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

using namespace std;
using namespace reco;
using namespace pixeltrackfitting;

PixelTrackCleanerBySharedHits::PixelTrackCleanerBySharedHits( const edm::ParameterSet& cfg)
{}

PixelTrackCleanerBySharedHits::~PixelTrackCleanerBySharedHits()
{}

namespace {
  inline
  bool recHitsAreEqual(const TrackingRecHit *recHit1, const TrackingRecHit *recHit2) {
    if (recHit1->geographicalId() != recHit2->geographicalId()) return false;
    LocalPoint pos1 = recHit1->localPosition();
    LocalPoint pos2 = recHit2->localPosition();
    return ((pos1.x() == pos2.x()) && (pos1.y() == pos2.y()));
  }
  
}

TracksWithRecHits PixelTrackCleanerBySharedHits::cleanTracks(const TracksWithRecHits & trackHitPairs,
							     const TrackerTopology *tTopo)
{
  typedef std::vector<const TrackingRecHit *> RecHits;
  vector<TrackWithRecHits> cleanedTracks;

  LogDebug("PixelTrackCleanerBySharedHits") << "Cleanering tracks" << "\n";
  unsigned int size = trackHitPairs.size();
  if (size == 0) return cleanedTracks;

  bool trackOk[size];
  for (auto i = 0U; i < size; i++) trackOk[i] = true;

  for (auto iTrack1 = 0U; iTrack1 < size; iTrack1++) {
    auto track1 = trackHitPairs[iTrack1].first;
    const RecHits& recHits1 = trackHitPairs[iTrack1].second;

    if (!trackOk[iTrack1]) continue;

    for (auto iTrack2 = iTrack1 + 1U; iTrack2 < size; iTrack2++)
    {
      if ( !trackOk[iTrack2]) continue; 

      auto track2 = trackHitPairs[iTrack2].first;
      const RecHits& recHits2 = trackHitPairs[iTrack2].second;

      auto commonRecHits = 0U;
      for (auto iRecHit1 = 0U; iRecHit1 < recHits1.size(); iRecHit1++) {
        for (auto iRecHit2 = 0U; iRecHit2 < recHits2.size(); iRecHit2++) {
          if (recHitsAreEqual(recHits1[iRecHit1], recHits2[iRecHit2])) { commonRecHits++; break;} // if a hit is common, no other can be the same!
        }
	if (commonRecHits > 1) break;
      }
      
      if (commonRecHits > 1) {
	if (track1->pt() > track2->pt()) trackOk[iTrack2] = false;
	else { trackOk[iTrack1] = false; break;}
      }

    }
  }

  for (auto i = 0U; i < size; i++)
  {
    if (trackOk[i]) cleanedTracks.push_back(trackHitPairs[i]);
    else delete trackHitPairs[i].first;
  }
  return cleanedTracks;
}
