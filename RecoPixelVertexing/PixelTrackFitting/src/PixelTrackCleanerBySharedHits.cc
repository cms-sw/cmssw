#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerBySharedHits.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

using namespace std;
using namespace reco;
using namespace pixeltrackfitting;

PixelTrackCleanerBySharedHits::PixelTrackCleanerBySharedHits(bool useQuadrupletAlgo):
  PixelTrackCleaner(true), // to mark this as fast algo
  useQuadrupletAlgo_(useQuadrupletAlgo)
{}

PixelTrackCleanerBySharedHits::~PixelTrackCleanerBySharedHits()
{}


void PixelTrackCleanerBySharedHits::cleanTracks(TracksWithTTRHs & trackHitPairs) const 
{

  LogDebug("PixelTrackCleanerBySharedHits") << "Cleanering tracks" << "\n";
  unsigned int size = trackHitPairs.size();
  if (size <= 1) return;

  auto kill = [&](unsigned int i) { delete trackHitPairs[i].first; trackHitPairs[i].first=nullptr;};

  for (auto iTrack1 = 0U; iTrack1 < size; iTrack1++) {

    auto track1 = trackHitPairs[iTrack1].first;
    if (!track1) continue;

    auto const & recHits1 = trackHitPairs[iTrack1].second;
    auto s1 = recHits1.size();
    for (auto iTrack2 = iTrack1 + 1U; iTrack2 < size; iTrack2++)
    {
      auto track2 = trackHitPairs[iTrack2].first;
      if (!track2) continue;
      auto const & recHits2 = trackHitPairs[iTrack2].second;
      auto s2 = recHits2.size();
      auto f2=0U;
      auto commonRecHits = 0U;
      for (auto iRecHit1 = 0U; iRecHit1 < s1; ++iRecHit1) {
        for (auto iRecHit2 = f2; iRecHit2 < s2; ++iRecHit2) {
          if (recHits1[iRecHit1] == recHits2[iRecHit2]) { ++commonRecHits; f2=iRecHit2+1; break;} // if a hit is common, no other can be the same!
        }
	if (commonRecHits > 1) break;
      }

      auto cleanTrack = [&](){
	if (track1->pt() > track2->pt()) { kill(iTrack2); return false; }
	kill(iTrack1);
        return true;
      };

      if(useQuadrupletAlgo_) {
        if(commonRecHits >= 1) {
          if     (s1 > s2) kill(iTrack2);
          else if(s1 < s2) { kill(iTrack1); break;}
          else if(s1 == 3) { if(cleanTrack()) break; } // same number of hits
          else if(commonRecHits > 1) { if(cleanTrack()) break; }// same number of hits, size != 3 (i.e. == 4)
        }
      }
      else if (commonRecHits > 1) {
        if(cleanTrack()) break;
      }
    }
  }

  trackHitPairs.erase(std::remove_if(trackHitPairs.begin(),trackHitPairs.end(),[&](TrackWithTTRHs & v){ return nullptr==v.first;}),trackHitPairs.end());
}
