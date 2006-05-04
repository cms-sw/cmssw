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
 * $Date: 2006/04/20 17:11:08 $
 * $Revision: 1.5 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"

/// Constructor
DTChamberRecSegment2D::DTChamberRecSegment2D(const DTChamberId id):DTRecSegment2D(id){}

/// c'tor from hits
DTChamberRecSegment2D::DTChamberRecSegment2D(const DTChamberId id, const std::vector<DTRecHit1D>& hits): 
  DTRecSegment2D(id,hits){}

/// complete constructor
DTChamberRecSegment2D::DTChamberRecSegment2D(const DTChamberId id, 
					     LocalPoint &position, LocalVector &direction,
					     AlgebraicSymMatrix & covMatrix, double &chi2, 
					     std::vector<DTRecHit1D> &hits1D):
  DTRecSegment2D(id, position, direction, covMatrix, chi2, hits1D){}

DTChamberRecSegment2D* DTChamberRecSegment2D::clone() const { 
  return new DTChamberRecSegment2D(*this);
}
  
DTChamberId DTChamberRecSegment2D::chamberId() const {
  return DTChamberId(theDetId.rawId());
}
