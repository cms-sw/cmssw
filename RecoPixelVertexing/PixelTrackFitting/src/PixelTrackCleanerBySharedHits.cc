#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerBySharedHits.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include <boost/function_output_iterator.hpp>

using namespace std;
using namespace reco;
using namespace pixeltrackfitting;

PixelTrackCleanerBySharedHits::PixelTrackCleanerBySharedHits( const edm::ParameterSet& cfg)
{fast=true;}

PixelTrackCleanerBySharedHits::~PixelTrackCleanerBySharedHits()
{}

namespace {
  inline
  bool recHitsLess(const TrackingRecHit *recHit1, const TrackingRecHit *recHit2) {
    if (recHit1->geographicalId() < recHit2->geographicalId()) return true;
    if (recHit1->geographicalId() == recHit2->geographicalId()) return recHit1 < recHit2;
    return false;
  }
}

void PixelTrackCleanerBySharedHits::cleanTracks(TracksWithTTRHs & trackHitPairs,
                                        const TrackerTopology *tTopo) const 
{

  LogDebug("PixelTrackCleanerBySharedHits") << "Cleanering tracks" << "\n";
  unsigned int size = trackHitPairs.size();
  if (size <= 1) return;

  auto kill = [&](unsigned int i) { delete trackHitPairs[i].first; trackHitPairs[i].first=nullptr;};

  for (auto iTrack1 = 0U; iTrack1 < size; iTrack1++) {

    auto track1 = trackHitPairs[iTrack1].first;
    if (!track1) continue;

    auto const & recHits1 = trackHitPairs[iTrack1].second;
    auto f1 = recHits1.data();
    auto s1 = recHits1.size();
    auto e1 = f1+s1;
    /*
    {
      auto f2 = recHits1.data();
      auto s2 = recHits1.size();
      auto e2 = f2+s1;
      auto commonRecHits = 0U;
      std::set_intersection(f1,e1,f2,e2,
                            boost::make_function_output_iterator([&](const TrackingRecHit *) {++commonRecHits;}),recHitsLess);
      assert(commonRecHits==s2);
    }
    */

    // for (auto iRecHit1 = 1U; iRecHit1 < s1; ++iRecHit1) assert(recHitsLess(recHits1[iRecHit1-1],recHits1[iRecHit1]));

    for (auto iTrack2 = iTrack1 + 1U; iTrack2 < size; iTrack2++)
    {
      auto track2 = trackHitPairs[iTrack2].first;
      if (!track2) continue;
      auto const & recHits2 = trackHitPairs[iTrack2].second;
      auto f2 = recHits2.data();
      auto s2 = recHits2.size();
      auto e2 = f2+s2;
      auto commonRecHits = 0U;
      std::set_intersection(f1,e1,f2,e2,
                            boost::make_function_output_iterator([&](const TrackingRecHit *) {++commonRecHits;}),
                            recHitsLess);
      
      if (commonRecHits > 1) {
	if (track1->pt() > track2->pt()) kill(iTrack2);
	else { kill(iTrack1); break;}
      }

    }
  }

  trackHitPairs.erase(std::remove_if(trackHitPairs.begin(),trackHitPairs.end(),[&](TrackWithTTRHs & v){ return nullptr==v.first;}),trackHitPairs.end());
}
