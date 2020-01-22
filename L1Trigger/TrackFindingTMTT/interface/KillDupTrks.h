#ifndef __KILLDUPTRKS_H__
#define __KILLDUPTRKS_H__

#include <cstddef>
#include <vector>
#include <algorithm>
#include <functional>
#include <utility>
#include <iostream>

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <gsl/gsl_fit.h>

using namespace std;

/**
*  Kill duplicate reconstructed tracks.
*  e.g. Those sharing many hits in common.
*  
*  Currently this is intended to run only on tracks found within a single (eta,phi) sector.
*  It includes a naive algorithms from Ian (dupTrkAlg = 1) & more sophisticated ones from Ivan (dupTrkAlg > 1).
*  The class is implemented inside L1Trigger/TrackFindingTMTT/interface/KillDupTrks.icc
*  
*  The template class "T" can be any class inheriting from L1trackBase.
* 
*  -------------------------------------------------------------------------------------------
*   GENERAL INFO ABOUT THE FILTER ALGORITHMS DEFINED IN THE CLASS.
*   Some of these algorithms are designed to work on r-phi L1track2D tracks, and some on r-z 
*   L1track2D tracks. Others work on L1tracks3D.
*  -------------------------------------------------------------------------------------------
*/

namespace TMTT {

class L1trackBase;
class L1track2D;
class L1track3D;
class L1fittedTrack;

template <class T> class KillDupTrks {

public:

  KillDupTrks()
	{
    // Check that classed used as template "T" inherits from class L1trackBase.
    static_assert(std::is_base_of<L1trackBase, T>::value, "KillDupTrks ERROR: You instantiated this with a template class not inheriting from L1trackBase!");
  }

  ~KillDupTrks() {}

  /**
  *  Make available cfg parameters & specify which algorithm is to be used for duplicate track removal.
  */
  void init(const Settings* settings, unsigned int dupTrkAlg);

  /**
  *  Eliminate duplicate tracks from the input collection, and so return a reduced list of tracks.
  */
  vector<T> filter(const vector<T>& vecTracks) const;

private:

  /**
  *  Implementing "inverse" OSU algorithm, check for stubs in common,
  *  keep largest candidates if common stubs in N or more layers (default 5 at present), both if equal
  *  Implementing "inverse" OSU algorithm, check for stubs in common,
  *  keep largest candidates if common stubs in N or more layers (default 5 at present), both if equal
  */
  vector<T> filterAlg8(const vector<T>& vecTracks) const;

  /** Implementing "inverse" OSU algorithm, check for layers in common, reverse order as per Luis's suggestion
   * Comparison window of up to 6
   * Modified version of Algo23, looking for layers in common as in Algo8
   * Check if N or more common layers (default 5 at present)
   * Then keep candidate with most stubs, use |q/pT| as tie-break, finally drop "latest" if still equal
   */
vector<T> filterAlg25(const vector<T>& vecTracks) const;

  /**
  *  Prints out a consistently formatted formatted report of killed duplicate track
  */
  void printKill(unsigned alg, unsigned dup, unsigned cand, T dupTrack, T candTrack) const;

  /**
  * Counts candidate layers with stubs in common
  */
  unsigned int layerMatches(std::vector< std::pair<unsigned int, unsigned int> >* iStubs,
			    std::vector< std::pair<unsigned int, unsigned int> >* jStubs) const;
private:

  const Settings *settings_; // Configuration parameters.

  unsigned int dupTrkAlg_; // Specifies choice of algorithm for duplicate track removal.
  unsigned int dupTrkMinCommonHitsLayers_;  // Min no of matched stubs & layers to keep smaller cand
};

}
//=== Include file which implements all the functions in the above class.
#include "L1Trigger/TrackFindingTMTT/interface/KillDupTrks.icc"

#endif

