#ifndef DTSLRecSegment2D_H
#define DTSLRecSegment2D_H

/** \class DTSLRecSegment2D
 *
 *  a 2D (x, dx/dz) segment in a DT superlayer.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

class DTSLRecSegment2D: public DTRecSegment2D{
public:
  /// Constructor
  DTSLRecSegment2D(){};

  /// c'tor from hits
  DTSLRecSegment2D(const DTSuperLayerId id, const std::vector<DTRecHit1D>& hits);
  
  /// complete constructor
  DTSLRecSegment2D(const DTSuperLayerId id, 
		   LocalPoint &position, LocalVector &direction,
		   AlgebraicSymMatrix & covMatrix, double &chi2, 
		   std::vector<DTRecHit1D> &hits1D);

  /// Destructor
  ~DTSLRecSegment2D() override{};

  // Operations

  /// The clone method needed by the clone policy
  DTSLRecSegment2D* clone() const override;
  
  /// The id of the superlayer on which reside the segment
  DTSuperLayerId superLayerId() const;

  /// The id of the chamber on which reside the segment
  DTChamberId chamberId() const;

private:
  friend class DTSegmentUpdator;
  
protected:

private:

};
#endif

