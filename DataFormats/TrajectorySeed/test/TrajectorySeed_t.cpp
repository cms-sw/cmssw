#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include<vector>


int main() {

  typedef std::vector<TrajectorySeed> TV;

  // absurd: just for test
  TrajectorySeed::recHitContainer c; c.reserve(1000);
  for (int i=0; i!=100; ++i) 
    c.push_back(new SiStripMatchedRecHit2D);

  for (int j=0; j!=1000; ++j;) {
    TV v;
    for (int i=0; i!=1000; ++i) 
      v.push_back(TrajectorySeed(PTrajectoryStateOnDet(), c, anyDirection));
  }

  return 0;
  
}
