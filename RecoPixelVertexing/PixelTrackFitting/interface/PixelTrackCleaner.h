#ifndef PixelTrackCleaner_H
#define PixelTrackCleaner_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

/*
PixelTrackCleaner:

Discards tracks with more than one common recHit.

*/

using namespace std;
using namespace reco;

typedef pair<reco::Track*, std::vector<const TrackingRecHit *> > TrackHitsPair;

class PixelTrackCleaner {

public:

	PixelTrackCleaner();

	vector<TrackHitsPair> cleanTracks(vector<TrackHitsPair> trackHitPairs);
  void cleanTrack();
  bool RecHitsAreEqual(const TrackingRecHit *recHit1, const TrackingRecHit *recHit2);

private:

  vector<bool> trackOk;
  reco::Track *track1, *track2;
  int iTrack1, iTrack2;

};

#endif
