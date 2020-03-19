#ifndef __KILLDUPFITTRKS_H__
#define __KILLDUPFITTRKS_H__

#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/KillDupTrks.h"

#include <vector>
#include <iostream>

using namespace std;

/**
*  Kill duplicate fitted tracks.
*  
*  Currently this is intended to run only on tracks found within a single (eta,phi) sector.
*
*  N.B. Duplicate track removal algorithms that can only be run on fitted tracks are implemented
*  here, whilst those that can also be run on the L1track2D or L1track3D collections are instead 
*  implemented inside class KillDupTrks.
*/

namespace TMTT {

class Settings;

class KillDupFitTrks {

public:

  KillDupFitTrks() {}

  ~KillDupFitTrks() {}

  /**
  *  Make available cfg parameters & specify which algorithm is to be used for duplicate track removal.
  */
  void init(const Settings* settings, unsigned int dupTrkAlg);

  /**
  *  Eliminate duplicate tracks from the input collection, and so return a reduced list of tracks.
  */
  vector<L1fittedTrack> filter(const vector<L1fittedTrack>& vecTracks) const;

private:

 /**
   * Duplicate removal algorithm designed to run after the track helix fit, which eliminates duplicates simply 
   * by requiring that the fitted (q/Pt, phi0) of the track correspond to the same HT cell in which the track
   * was originally found by the HT.
   */
  vector<L1fittedTrack> filterAlg50(const vector<L1fittedTrack>& tracks) const;

  /**
    * Duplicate removal algorithm designed to run after the track helix fit, which eliminates duplicates  
    * simply by requiring that no two tracks should have fitted (q/Pt, phi0) that correspond to the same HT
    * cell. If they do, then only the first to arrive is kept.
  */
  vector<L1fittedTrack> filterAlg51(const vector<L1fittedTrack>& tracks) const;

  /**
   * Other duplicate track removal algorithms are available in class KillDupTrks, which this class
   * can call.
   */

  // Debug printout of which tracks are duplicates.
  void printDuplicateTracks(const vector<L1fittedTrack>& tracks) const;

private:

  const Settings *settings_; // Configuration parameters.
  unsigned int dupTrkAlg_; // Specifies choice of algorithm for duplicate track removal.
  KillDupTrks<L1fittedTrack> killDupTrks_;  // Contains duplicate removal algorithms common to all track types.
};

}

#endif

