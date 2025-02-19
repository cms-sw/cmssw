#include "RecoTracker/CkfPattern/interface/SeedCleanerBySharedInput.h"

bool SeedCleanerBySharedInput::good(const TrajectorySeed *seed) {
    typedef TrajectorySeed::const_iterator SI;
    typedef Trajectory::RecHitContainer::const_iterator TI;
    TrajectorySeed::range range = seed->recHits();
    SI first = range.first, curr = range.first, last = range.second;
//     for (std::vector<Trajectory>::const_iterator trj = trajectories->begin(),
//             trjEnd =trajectories->end();
//             trj != trjEnd; ++trj) {
// Try reverse direction (possible gain in speed)
    for (std::vector<Trajectory>::const_reverse_iterator trj = trajectories->rbegin(),
            trjEnd =trajectories->rend();
            trj != trjEnd; ++trj) {
        Trajectory::RecHitContainer hits = trj->recHits();
        TI ts = hits.begin(), tc = ts, te = hits.end();
        for (curr = first; curr < last; ++curr) {
            bool found = false;
            for (TI it = tc; it != te; ++it) {
	      if ( curr->sharesInput((**it).hit(),TrackingRecHit::all) ) {
                    tc = it; found = true; break;         
                }
            }
            if (found == false) {
                for (TI it = ts; it != tc; ++it) {
		  if ( curr->sharesInput((**it).hit(),TrackingRecHit::all) ) {
                        tc = it; found = true; break;         
                    }
                }
                if (found == false) break;
            }
        }
        if (curr == last)  return false;   
    }
    return true;
}
