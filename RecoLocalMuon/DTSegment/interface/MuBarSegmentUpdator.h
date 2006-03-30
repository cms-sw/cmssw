#ifndef MUBARSEGMENTUPDATOR_H
#define MUBARSEGMENTUPDATOR_H

/** \class MuBarSegmentUpdator
 *
 * Update a segment by improving the hits thanks to the refined knowledge of
 * impact angle and position (also along the wire).
 *
 * \author Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 * $date   03/09/2003 11:55:59 CEST $
 *
 * Modification:
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */
#include "Utilities/UI/interface/SimpleConfigurable.h"
#include "ClassReuse/GeomVector/interface/LocalPoint.h"
#include "ClassReuse/GeomVector/interface/LocalVector.h"
#include "ClassReuse/GeomVector/interface/GlobalPoint.h"
#include "ClassReuse/GeomVector/interface/GlobalVector.h"
#include "CommonDet/DetGeometry/interface/AlgebraicObjects.h"
class LinearFit;
class MuBarHitBaseAlgo;
class MuBarSegmentCand;
class MuBarSegment2D;
class MuBarSegment;

/* C++ Headers */
#include <string>
#include <vector>

/* ====================================================================== */

/* Class MuBarSegmentUpdator Interface */

class MuBarSegmentUpdator{

  public:

/** Constructor */ 
    MuBarSegmentUpdator() ;

/** Destructor */ 
    ~MuBarSegmentUpdator() ;

/* Operations */ 
    /// recompute hits position and refit the segment4D
    void update(MuBarSegment* seg);

    /// recompute hits position and refit the segment2D
    void update(MuBarSegment2D* seg);

    /** do the linear fit on the hits of the segment candidate and update it.
     * Returns false if the segment candidate is not good() */
    bool fit(MuBarSegmentCand* seg);

    /** ditto for true segment: since the fit is applied on a true segment, by
     * definition the segment is "good", while it's not the case for just
     * candidates */
    void fit(MuBarSegment2D* seg);

    /** ditto for true segment 4D, the fit is done on either projection and then
     * the 4D direction and position is built. Since the fit is applied on a
     * true segment, by definition the segment is "good", while it's not the
     * case for just candidates */
    void fit(MuBarSegment* seg);

  private:
    void updateHits(MuBarSegment2D* seg);

    void updateHits(MuBarSegment2D* seg,
                    GlobalPoint gpos,
                    GlobalVector gdir,
                    int step=2);

    /// interface to LinearFit
    void fit(const vector<float>& x,
             const vector<float>& y, 
             const vector<float>& sigy,
             LocalPoint& pos,
             LocalVector& dir,
             AlgebraicSymMatrix& covMat,
             double& chi2);

    MuBarHitBaseAlgo* theAlgo;

    LinearFit* theFitter;

    static SimpleConfigurable<string> algoForDriftAndError;

  protected:

};
#endif // MUBARSEGMENTUPDATOR_H

