#ifndef DTRecHit_DTRecSegment2DPhi_h
#define DTRecHit_DTRecSegment2DPhi_h

/** \class DTRecSegment2DPhi
 *
 * A 2D segment for the DT system for Phi projection.
 * It's an intermediate data class between the normal DTRecSegment2D class and
 * the DTRecSegment4D. The difference wrt DTRecSegment2D is that the segments it
 * represents is build with the two phi SL. So this segment DOES not belong to
 * the SL (as DTRecSegment2D), but to the chamber (via a DTRecSegment4D).
 * A DTRecSegment4D has one of these objects, and so can access the full
 * information of the two projections.
 *
 * $Date: 2006/02/23 10:32:04 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"

/* Collaborating Class Declarations */

/* C++ Headers */

/* ====================================================================== */

/* Class DTRecSegment2DPhi Interface */

class DTRecSegment2DPhi : public DTRecSegment2D {

  public:

/// Constructor
    DTRecSegment2DPhi(const DetId& id) ;

/// Destructor
    virtual ~DTRecSegment2DPhi() ;

/* Operations */ 

  protected:

  private:

};
#endif // DTRecHit_DTRecSegment2DPhi_h

