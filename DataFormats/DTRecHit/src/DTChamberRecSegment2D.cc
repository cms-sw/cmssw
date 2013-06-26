/** \file
 *
 * $Date: 2007/08/02 05:56:33 $
 * $Revision: 1.4 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"

// c'tor from hits
DTChamberRecSegment2D::DTChamberRecSegment2D(const DTChamberId id, const std::vector<DTRecHit1D>& hits): 
  DTRecSegment2D(id,hits){}

// complete constructor
DTChamberRecSegment2D::DTChamberRecSegment2D(const DTChamberId id, 
					     LocalPoint &position, LocalVector &direction,
					     AlgebraicSymMatrix & covMatrix, double chi2, 
					     std::vector<DTRecHit1D> &hits1D):
  DTRecSegment2D(id, position, direction, covMatrix, chi2, hits1D){}

DTChamberRecSegment2D* DTChamberRecSegment2D::clone() const { 
  return new DTChamberRecSegment2D(*this);
}
  
DTChamberId DTChamberRecSegment2D::chamberId() const {
  return DTChamberId(geographicalId());
}
