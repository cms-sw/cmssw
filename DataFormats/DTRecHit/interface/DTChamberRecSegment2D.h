#ifndef DTRecHit_DTChamberRecSegment2D_h
#define DTRecHit_DTChamberRecSegment2D_h

/** \class DTChamberRecSegment2D
 *
 * A 2D segment for the DT system for Phi projection.
 * It's an intermediate data class between the normal DTSLRecSegment2D class and
 * the DTRecSegment4D. The difference wrt DTSLRecSegment2D is that the segments it
 * represents is build with the two phi SL. So this segment DOES not belong to
 * the SL (as DTSLRecSegment2D), but to the chamber (via a DTRecSegment4D).
 * A DTRecSegment4D has one of these objects, and so can access the full
 * information of the two projections.
 *
 * $Date: 2007/08/02 05:54:11 $
 * $Revision: 1.4 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

/* Collaborating Class Declarations */

/* C++ Headers */

/* ====================================================================== */

/* Class DTChamberRecSegment2D Interface */

class DTChamberRecSegment2D : public DTRecSegment2D {

 public:
  
  /// empty c'tor 
  DTChamberRecSegment2D() {}

  /// c'tor from hits
  DTChamberRecSegment2D(DTChamberId id, const std::vector<DTRecHit1D>& hits) ;
  
  /// complete constructor
  DTChamberRecSegment2D(DTChamberId id, 
			LocalPoint &position, LocalVector &direction,
			AlgebraicSymMatrix & covMatrix, double chi2, 
			std::vector<DTRecHit1D> &hits1D);
  
  /// Destructor
  virtual ~DTChamberRecSegment2D(){};

  /* Operations */ 

  /// The clone method needed by the clone policy
  virtual DTChamberRecSegment2D* clone() const;
  
  /// The id of the chamber on which reside the segment
  DTChamberId chamberId() const;

 private:
  // in DTSegmentCand, setPosition and setDirection can be used
  friend class DTSegmentCand; 
  friend class DTSegmentUpdator;
  void setChamberId(DTChamberId chId){ setId(chId);}

 protected:



};
#endif // DTRecHit_DTChamberRecSegment2D_h

