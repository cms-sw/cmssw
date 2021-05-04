#ifndef L1Trigger_TrackFindingTMTT_DupFitTrkKiller_h
#define L1Trigger_TrackFindingTMTT_DupFitTrkKiller_h

#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"

#include <list>
#include <iostream>

/**
*  Kill duplicate fitted tracks.
*  
*  Currently this is intended to run only on tracks found within a single (eta,phi) sector.
*/

namespace tmtt {

  class Settings;

  class DupFitTrkKiller {
  public:
    enum class DupAlgoName { None = 0, Algo1 = 1, Algo2 = 2 };

    /**
  *  Make available cfg parameters & specify which algorithm is to be used for duplicate track removal.
  */
    DupFitTrkKiller(const Settings* settings);

    /**
  *  Eliminate duplicate tracks from the input collection, and so return a reduced list of tracks.
  */
    std::list<const L1fittedTrack*> filter(const std::list<L1fittedTrack>& vecTracks) const;

  private:
    /**
   * Duplicate removal algorithm that eliminates duplicates 
   * by requiring that the fitted (q/Pt, phi0) of the track correspond to the same HT cell in which the track
   * was originally found by the HT.
   */
    std::list<const L1fittedTrack*> filterAlg1(const std::list<L1fittedTrack>& tracks) const;

    /**
    * Duplicate removal algorithm that  eliminates duplicates  
    * by requiring that no two tracks should have fitted (q/Pt, phi0) that correspond to the same HT
    * cell. If they do, then only the first to arrive is kept.
  */
    std::list<const L1fittedTrack*> filterAlg2(const std::list<L1fittedTrack>& tracks) const;

    // Debug printout of which tracks are duplicates.
    void printDuplicateTracks(const std::list<const L1fittedTrack*>& tracks) const;

  private:
    const Settings* settings_;  // Configuration parameters.
    DupAlgoName dupTrkAlg_;     // Specifies choice of algorithm for duplicate track removal.
  };

}  // namespace tmtt

#endif
