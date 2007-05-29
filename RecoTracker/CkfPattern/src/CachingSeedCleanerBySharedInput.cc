#include "RecoTracker/CkfPattern/interface/CachingSeedCleanerBySharedInput.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

void CachingSeedCleanerBySharedInput::init(const std::vector<Trajectory> *vect) { 
    theVault.clear(); theCache.clear();
}

void CachingSeedCleanerBySharedInput::done() { 
    //edm::LogInfo("CachingSeedCleanerBySharedInput") << " Calls: " << calls_ << ", Tracks: " << tracks_ <<", Comps: " << comps_ << " Vault: " << theVault.size() << ".";
    //calls_ = comps_ = tracks_ = 0;
    theVault.clear(); theCache.clear();

    std::vector<Trajectory::RecHitContainer> swapper;
    swapper.swap(theVault); // this should clean the vault even more
}



void CachingSeedCleanerBySharedInput::add(const Trajectory *trj) {
    typedef Trajectory::RecHitContainer::const_iterator TI;
    unsigned short idx = theVault.size();
    Trajectory::RecHitContainer hits = trj->recHits();
    theVault.push_back(hits);

    uint32_t detid;
    for (TI t = hits.begin(), te = hits.end(); t != te; ++t) {
        if ((*t)->isValid()) {
            detid = (*t)->geographicalId().rawId();
            if (detid) theCache.insert(std::pair<uint32_t, unsigned short>(detid, idx));
        }
    }
}

bool CachingSeedCleanerBySharedInput::good(const TrajectorySeed *seed) {
    typedef BasicTrajectorySeed::const_iterator SI;
    typedef Trajectory::RecHitContainer::const_iterator TI;
    BasicTrajectorySeed::range range = seed->recHits();

    SI first = range.first, last = range.second, curr;
    uint32_t detid = first->geographicalId().rawId();
    
    std::multimap<uint32_t, unsigned short>::const_iterator it, end = theCache.end();

    //calls_++;
    for (it = theCache.find(detid); (it != end) && (it->first == detid); ++it) {
        //tracks_++;
        TI ts = theVault[it->second].begin(), te = theVault[it->second].end();
        for (curr = first; curr != last; ++curr) {
            bool found = false;
            for (TI t = ts; t != te; ++t) {
                //comps_++;
                if ( curr->sharesInput((**t).hit(),TrackingRecHit::all) ) { found = true; break; }
            }
            if (!found) break;
        }
        if (curr == last) return false;
    }
    return true;
}
