#ifndef DTSegment_DTSegmentCleaner_h
#define DTSegment_DTSegmentCleaner_h

/** \file
 *
 *  Conflict solving and ghost suppression for DT segment candidates 
 *
 *  The candidates found by DTCombinatorialPatternReco are checked for
 *  conflicting segments, namely segment which shares hits with conflicting
 *  Left/Right ambuguity solution. Then check for ghost, i.e. segments which
 *  shares more than a given number of hits. In both cases, a selection is done
 *  retaining the segment with higher number of hits and best chi2, while the
 *  others are deleted.
 *
 * $Date: 2006/03/30 16:53:18 $
 * $Revision: 1.1 $
 * \author : Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* Base Class Headers */

/* Collaborating Class Declarations */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"

/* C++ Headers */
#include <vector>

/* ====================================================================== */

/* Class DTSegmentCleaner Interface */

class DTSegmentCleaner{

  public:

/* Constructor */ 
    DTSegmentCleaner(const edm::ParameterSet& pset) ;

/* Destructor */ 
    ~DTSegmentCleaner() ;

/* Operations */ 
    /// do the cleaning
    std::vector<DTSegmentCand*> clean(std::vector<DTSegmentCand*> inputCands) const ;

  private:
    /// solve the conflicts
    std::vector<DTSegmentCand*> solveConflict(std::vector<DTSegmentCand*> inputCands) const ;

    /// ghost  suppression
    std::vector<DTSegmentCand*> ghostBuster(std::vector<DTSegmentCand*> inputCands) const ;

    int nSharedHitsMax;

};
#endif // DTSegment_DTSegmentCleaner_h
