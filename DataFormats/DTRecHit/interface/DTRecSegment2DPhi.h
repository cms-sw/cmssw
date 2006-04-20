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
 * $Date: 2006/04/19 17:42:41 $
 * $Revision: 1.4 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

class DTChamber;
/* Collaborating Class Declarations */

/* C++ Headers */

/* ====================================================================== */

/* Class DTRecSegment2DPhi Interface */

class DTRecSegment2DPhi : public DTRecSegment2D {

 public:
  
  /// empty c'tor 
  DTRecSegment2DPhi() {}

  /// Constructor
  DTRecSegment2DPhi(const DetId& id) ;

  /// c'tor from hits
  DTRecSegment2DPhi(const DTChamberId& id, const std::vector<DTRecHit1D>& hits) ;
  
  /// Destructor
  virtual ~DTRecSegment2DPhi() ;

  /* Operations */ 

 private:

  // in DTSegmentCand, setPosition and setDirection can be used
  friend class DTSegmentCand; 

 protected:



};
#endif // DTRecHit_DTRecSegment2DPhi_h

