#include "RecoTracker/CkfPattern/interface/SeedCleanerByHitPosition.h"
#include "TrackingTools/TransientTrackingRecHit/interface/RecHitComparatorByPosition.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

void SeedCleanerByHitPosition::done() {
        // edm::LogInfo("SeedCleanerByHitPosition") << " Calls: " << calls_ << ", Tracks: " << tracks_ <<", Comps: " << comps_  << " Vault: " << trajectories->size() << ".";
        // calls_ = comps_ = tracks_ = 0;

        trajectories = 0; 
}
bool SeedCleanerByHitPosition::good(const TrajectorySeed *seed) {
    static RecHitComparatorByPosition comp;
    typedef TrajectorySeed::const_iterator SI;
    typedef Trajectory::RecHitContainer::const_iterator TI;
    TrajectorySeed::range range = seed->recHits();
    SI first = range.first, curr = range.first, last = range.second;
    //calls_++;
    for (std::vector<Trajectory>::const_iterator trj = trajectories->begin(),
            trjEnd =trajectories->end();
            trj != trjEnd; ++trj) {
        //tracks_++;
        Trajectory::RecHitContainer hits = trj->recHits();
        TI ts = hits.begin(), tc = ts, te = hits.end();
        for (curr = first; curr < last; ++curr) {
            bool found = false;
            for (TI it = tc; it != te; ++it) {
                //comps_++;
                if (comp.equals(&(*curr), &(**it))) {
                    tc = it; found = true; break;         
                }
            }
            if (found == false) {
                for (TI it = ts; it != tc; ++it) {
                    //comps_++;
                    if (comp.equals(&(*curr), &(**it))) {
                        tc = it; found = true; break;         
                    }
                }
                if (found == false) break;
            }
        }
        if (curr == last) return false;   
    }
    return true;
}
