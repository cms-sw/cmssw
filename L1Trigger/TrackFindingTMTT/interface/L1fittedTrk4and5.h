#ifndef __L1fittedTrk4and5_H__
#define __L1fittedTrk4and5_H__

#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <map>
//#include <utility>

using namespace std;

//=== This represents a fitted L1 track candidate found in 3 dimensions.
//===
//=== Since tracks fits can be done with 4 or 5 helix parameters, (where the 4 param case fixes d0 = 0),
//=== this class stores the results of both 4 and 5 parameter fits if available.
//===
//=== For both fits (4 and 5 param), it gives access to the fitted helix parameters & chi2 etc.
//=== It also gives access to the 3D hough-transform track candidate (L1track3D) on which the fit was run.
//=== This in turn tells one which stubs are on the track and which truth particle it is associated to, if any.
//===
//=== To get for example, the track params of the 5 param track fit, you can put in your analysis code:
//===    if (l1fittedTrk4and5.validL1fittedTrack(5) {
//===      const L1fittedTrack& trk5par = l1fittedTrk4and5.getL1fittedTrack(5); 
//===      float phi0 = trk5par.phi0();
//===    }

namespace TMTT {

class L1fittedTrk4and5 {

public:

  // Initialization. Make Hough transform (pre-fit!) track available.
  L1fittedTrk4and5(const L1track3D& trkHT) : trkHT_(trkHT) {}

  // You should call this to store the results of successful 4 or 5 parameter track fit.
  // Call it twice, if both 4 and 5 parameter track fits yielded valid tracks, specifying the results of 
  // each fit in turn (order irrelevant).
  // Call it only once if one of the two fits yielded a valid track and the other failed.
  // Don't call it if both fits failed.
  // (Track fits can fail if the fit removes too many stubs from the track because they have large residuals).

  void storeTrk(const L1fittedTrack& trk) {
    unsigned int nPar = trk.nHelixParam(); // 4 or 5 param helix fit used ?
    if ( trkMap_.find(nPar) == trkMap_.end() ) {
      // Only bother to store fitted track if it is valid.
      if (trk.accepted()) {
	trkMap_.insert( pair<unsigned int, L1fittedTrack>(nPar, trk) );
      }
    } else {
      throw cms::Exception("L1fittedTrk4and5:: You tried to store two tracks with the same number ofhelix parameters.");
    }    
  }

  ~L1fittedTrk4and5() {}

  // Check if the track fit was successful for the specified number of helix parameters.
  bool   validL1fittedTrack(unsigned int nPar) const  {
    return ( trkMap_.find(nPar) != trkMap_.end() ); // Fit successful if track stored in map.
  }

  // Get the fitted track, specifying if 4 or 5 parameter results wanted.   
  // IMPORTANT: You MUST check that validFittedTrack returns "true" before calling this function.
  const  L1fittedTrack& getL1fittedTrack(unsigned int nPar) const  {
    if (this->validL1fittedTrack( nPar )) {
      return trkMap_.at(nPar);
    } else {
      throw cms::Exception("L1fittedTrk4and5:: You tried to access a fitted track for which the fit result was not available.");
    }
  }

  // Give direct access to Hough transform track candidate (pre-fit).
  // This is usually also available via getL1fittedTrack().getL1track3D().
  // However, the latter will not work if neither 4 or 5 parameter helix
  // fits gave valid tracks.
  const L1track3D& getL1track3D() const {return trkHT_;}

private:

  // Fitted tracks obtained with 4 or 5 parameter helix fits.
  // The first parameter in the map is the number of parameters used for the fit.
  map<unsigned int, L1fittedTrack> trkMap_;

  // Hough transform (pre-fit) track candidate.
  L1track3D trkHT_;
};

}

#endif

