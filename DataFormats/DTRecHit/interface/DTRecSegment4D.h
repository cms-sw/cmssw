#ifndef DTRecHit_DTRecSegment4D_h
#define DTRecHit_DTRecSegment4D_h

/** \class DTRecSegment4D
 *
 * 4 parameters RecHits for MuonBarrel DT
 *
 * $Date: 2006/02/23 10:32:04 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/RecSegment4D.h"
//#include "DataFormats/MuonDetId/interface/DTChamberId.h"

/* Collaborating Class Declarations */
class DTRecSegment2DPhi;
class DTRecSegment2D;

/* C++ Headers */
#include <iosfwd>

/* ====================================================================== */

/* Class DTRecSegment4D Interface */

class DTRecSegment4D : public RecSegment4D{

 public:
  friend class DTSegmentUpdator;
  /// Constructor
  /// empty constructor
  DTRecSegment4D(){}
  
  DTRecSegment4D(DTRecSegment2DPhi* phiSeg, DTRecSegment2D* zedSeg) ;
  DTRecSegment4D(DTRecSegment2DPhi* phiSeg) ;
  DTRecSegment4D(DTRecSegment2D* zedSeg) ;

  /// Destructor
  ~DTRecSegment4D() ;

  /* Operations */ 

  virtual DTRecSegment4D* clone() const { return new DTRecSegment4D(*this);}

  
  AlgebraicVector parameters() const ;
  AlgebraicSymMatrix parametersError() const ;


  /// local position in Chamber frame
  virtual LocalPoint localPosition() const { return thePosition;}

  /// local position error in Chamber frame
  virtual LocalError localPositionError() const ;

  /// the local direction in Chamber frame
  virtual LocalVector localDirection() const { return theDirection;}

  /// the local direction error (xx,xy,yy) in Chamber frame: only xx is not 0.
  virtual LocalError localDirectionError() const ;

  /// the chi2 of the fit
  virtual double chi2() const ;
  
  /// return the DOF of the segment 
  virtual int degreesOfFreedom() const ;

  /// Access to component RecHits (if any)
  virtual std::vector<const TrackingRecHit*> recHits() const ;

  /// Non-const access to component RecHits (if any)
  virtual std::vector<TrackingRecHit*> recHits() ;

  /// the id 
  virtual DetId geographicalId() const { return theDetId; }
  
  /// has the Phi projection?
  bool hasPhi() const {return !(thePhiSeg==0);}
    
  /// has the Z projection?
  bool hasZed() const {return !(theZedSeg==0);}
    
  /// the superPhi segment
  DTRecSegment2DPhi* phiSegment() const {return thePhiSeg;}
    
  /// the Z segment
  DTRecSegment2D* zSegment() const {return theZedSeg;}
    
  /// set position
  void setPosition(LocalPoint pos) { thePosition = pos; }

  /// set direction
  void setDirection(LocalVector dir) { theDirection = dir; }

  /// set covariance matrix
  void setCovMatrix(AlgebraicSymMatrix mat) { theCovMatrix = mat; }

  // /// The id of the chamber on which reside the segment
  // DTChamberId chamberId() const;
    
 private:
  LocalPoint thePosition;   // in chamber frame
  LocalVector theDirection; // in chamber frame

  void setCovMatrixForZed(const LocalPoint& posZInCh);
    
  /// mat[0][0]=sigma (dx/dz)
  /// mat[1][1]=sigma (dy/dz)
  /// mat[2][2]=sigma (x)
  /// mat[3][3]=sigma (y)
  /// mat[0][2]=cov(dx/dz,x)
  /// mat[1][3]=cov(dy/dz,y)
  AlgebraicSymMatrix theCovMatrix; // the covariance matrix

  DTRecSegment2DPhi* thePhiSeg;
  DTRecSegment2D* theZedSeg;

  AlgebraicMatrix theProjMatrix;  // the projection matrix
  int theDimension; // the dimension of this rechit

  DetId theDetId;           // Id of the det this seg belongs

};

std::ostream& operator<<(std::ostream& os, const DTRecSegment4D& seg);

#endif // DTRecHit_DTRecSegment4D_h

