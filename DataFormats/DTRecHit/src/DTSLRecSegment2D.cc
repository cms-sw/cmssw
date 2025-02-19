/** \file
 *
 *  $Date: 2007/08/02 06:00:05 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"

/// c'tor from hits
DTSLRecSegment2D::DTSLRecSegment2D(const DTSuperLayerId id, const std::vector<DTRecHit1D>& hits):
  DTRecSegment2D(id,hits){}
  
/// complete constructor
DTSLRecSegment2D::DTSLRecSegment2D(const DTSuperLayerId id, 
		 LocalPoint &position, LocalVector &direction,
		 AlgebraicSymMatrix & covMatrix, double &chi2, 
				   std::vector<DTRecHit1D> &hits1D):
  DTRecSegment2D(id, position, direction, covMatrix, chi2, hits1D){}


/// The clone method needed by the clone policy
DTSLRecSegment2D* DTSLRecSegment2D::clone() const { 
  return new DTSLRecSegment2D(*this);
}

/// The id of the superlayer on which reside the segment
DTSuperLayerId DTSLRecSegment2D::superLayerId() const{
  return DTSuperLayerId(geographicalId());
}

/// The id of the chamber on which reside the segment
DTChamberId DTSLRecSegment2D::chamberId() const{
  return superLayerId().chamberId();
}
