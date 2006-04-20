#ifndef DTRecHit_DTRecSegment2D_h
#define DTRecHit_DTRecSegment2D_h

/** \class DTRecSegment2D
 *
 * 2D Segments for the muon barrel system.
 * 2D means that this segment has information about position and direction in
 * one projection (r-phi or r-theta/zeta).
 *  
 * $Date: 2006/04/19 15:05:25 $
 * $Revision: 1.4 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/RecSegment2D.h"

/* Collaborating Class Declarations */
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
class DTSegmentUpdator;

/* C++ Headers */
#include <iosfwd>

/* ====================================================================== */

/* Class DTRecSegment2D Interface */

class DTRecSegment2D : public RecSegment2D {

  public:
  
  /// Constructor
  /// empty c'tor needed by POOL (I guess)
  DTRecSegment2D() {}
  /// c'tor with no hits
  DTRecSegment2D(const DetId& id) ;
  
  /// c'tor from hits
  DTRecSegment2D(const DetId& id, const std::vector<DTRecHit1D>& hits) ;
  
  /// complete constructor
  DTRecSegment2D(const DetId &id, 
		 LocalPoint &position, LocalVector &direction,
		 AlgebraicSymMatrix & covMatrix, double &chi2, 
		 std::vector<DTRecHit1D> &hits1D);
  
  /// Destructor
  virtual ~DTRecSegment2D() ;
  
  /* Operations */ 
  /// The clone method needed by the clone policy
  virtual DTRecSegment2D* clone() const { return new DTRecSegment2D(*this);}

  /// the id 
  virtual DetId geographicalId() const { return theDetId; }

  /// local position in SL frame
  virtual LocalPoint localPosition() const {return thePosition; }
  
  /// local position error in SL frame
  virtual LocalError localPositionError() const ;
  
  /// the local direction in SL frame
  virtual LocalVector localDirection() const { return theDirection; }

  /// the local direction error (xx,xy,yy) in SL frame: only xx is not 0.
  virtual LocalError localDirectionError() const;

  /// the chi2 of the fit
  virtual double chi2() const { return theChi2; }
    
  /// return the DOF of the segment 
  virtual int degreesOfFreedom() const ;

  /// Access to component RecHits (if any)
  virtual std::vector<const TrackingRecHit*> recHits() const ;

  /// Non-const access to component RecHits (if any)
  virtual std::vector<TrackingRecHit*> recHits() ;

  /// Access to specific components
  std::vector<DTRecHit1D> specificRecHits() const ;
  
  /// The id of the chamber on which reside the segment
  DTChamberId chamberId() const {return superLayerId().chamberId();}

  /// the Covariance Matrix 
  AlgebraicSymMatrix covMatrix() const {return theCovMatrix;}

 protected:
  void setPosition(const LocalPoint& pos);
  void setDirection(const LocalVector& dir);
  void setCovMatrix(const AlgebraicSymMatrix& cov);
  void setChi2(const double& chi2);
  
 private:
  friend class DTSegmentUpdator;
  void update(std::vector<DTRecHit1D> & updatedRecHits);
  void setT0(const double& t0);

  /// The id of the superlayer on which reside the segment
  DTSuperLayerId superLayerId() const{return DTSuperLayerId(theDetId.rawId());}
  
 protected:
  DetId theDetId;           // Id of the det this seg belongs
 private:
  LocalPoint  thePosition;  // in SL frame
  LocalVector theDirection; // in SL frame

  /// mat[0][0]=sigma (dx/dz)
  /// mat[1][1]=sigma (x)
  /// mat[0][1]=cov(dx/dz,x)
  AlgebraicSymMatrix theCovMatrix; // the covariance matrix

  double theChi2;           // chi2 of the fit
 
 private:
  double theT0;             // T0 as coming from the fit
  std::vector<DTRecHit1D> theHits; // the hits with defined R/L
};

std::ostream& operator<<(std::ostream& os, const DTRecSegment2D& seg);

#endif // DTRecHit_DTRecSegment2D_h

