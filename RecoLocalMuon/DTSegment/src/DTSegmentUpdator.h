#ifndef DTSegment_DTSegmentUpdator_h
#define DTSegment_DTSegmentUpdator_h

/** \class DTSegmentUpdator
 *
 * Perform linear fit and hits update for DT segments.
 * Update a segment by improving the hits thanks to the refined knowledge of
 * impact angle and position (also along the wire) and perform linear fit on
 * improved hits.
 *
 * $Date: 2006/03/30 16:53:18 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"
//#include "DataFormats/DTRecHit/interface/DTRecSegment.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
class DTLinearFit;
class DTRecHitBaseAlgo;

namespace edm{class EventSetup;}

/* C++ Headers */
#include <vector>

/* ====================================================================== */

/* Class DTSegmentUpdator Interface */

class DTSegmentUpdator{

  public:

/// Constructor
    DTSegmentUpdator(const edm::ParameterSet& config) ;

/// Destructor
    ~DTSegmentUpdator() ;

/* Operations */ 

    /** do the linear fit on the hits of the segment candidate and update it.
     * Returns false if the segment candidate is not good() */
    bool fit(DTSegmentCand* seg);

    /** ditto for true segment: since the fit is applied on a true segment, by
     * definition the segment is "good", while it's not the case for just
     * candidates */
    void fit(DTRecSegment2D* seg);

    /** ditto for true segment 4D, the fit is done on either projection and then
     * the 4D direction and position is built. Since the fit is applied on a
     * true segment, by definition the segment is "good", while it's not the
     * case for just candidates */
    //void fit(DTRecSegment* seg);

    /// recompute hits position and refit the segment4D
    //void update(DTRecSegment* seg);

    /// recompute hits position and refit the segment2D
    void update(DTRecSegment2D* seg);

    /// set the setup
    void setES(const edm::EventSetup& setup);

  protected:

  private:
    DTLinearFit* theFitter; // the linear fitter
    DTRecHitBaseAlgo* theAlgo; // the algo for hit reconstruction
    edm::ESHandle<DTGeometry> theGeom; // the geometry

  private:

    void updateHits(DTRecSegment2D* seg);

    void updateHits(DTRecSegment2D* seg,
                    GlobalPoint gpos,
                    GlobalVector gdir,
                    int step=2);

    /// interface to LinearFit
    void fit(const std::vector<float>& x,
             const std::vector<float>& y, 
             const std::vector<float>& sigy,
             LocalPoint& pos,
             LocalVector& dir,
             AlgebraicSymMatrix& covMat,
             double& chi2);

};
#endif // DTSegment_DTSegmentUpdator_h
