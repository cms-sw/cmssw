#ifndef DTRecHit_DTRecSegment4D_h
#define DTRecHit_DTRecSegment4D_h

/** \class DTRecSegment4D
 *
 * 4 parameters RecHits for MuonBarrel DT
 *
 * $Date: 2006/05/02 07:08:42 $
 * $Revision: 1.5 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/RecSegment4D.h"
//#include "DataFormats/MuonDetId/interface/DTChamberId.h"

/* Collaborating Class Declarations */
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"

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
  
  //FIXME do only one constructor!
  DTRecSegment4D(const DTChamberRecSegment2D& phiSeg, const DTSLRecSegment2D& zedSeg, const LocalPoint& posZInCh, const LocalVector& dirZInCh);
  DTRecSegment4D(const DTChamberRecSegment2D& phiSeg);
  DTRecSegment4D(const DTSLRecSegment2D& zedSeg, const LocalPoint& posZInCh, const LocalVector& dirZInCh);
  //


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
  
  /// has the Phi projection? //FIXME, was right with the check on the pointers
  bool hasPhi() const {return (thePhiSeg.specificRecHits().size()!=0);}
  
  /// has the Z projection? //FIXME, was right with the check on the pointers
  bool hasZed() const {return (theZedSeg.specificRecHits().size()!=0);}
  
  /// the superPhi segment 
  const DTChamberRecSegment2D *phiSegment() const {return &thePhiSeg;}
    
  /// the Z segment
  const DTSLRecSegment2D *zSegment() const {return &theZedSeg;}
    
  /// set position
  void setPosition(LocalPoint pos) { thePosition = pos; }

  /// set direction
  void setDirection(LocalVector dir) { theDirection = dir; }

  /// set covariance matrix
  void setCovMatrix(AlgebraicSymMatrix mat) { theCovMatrix = mat; }

  /// The id of the chamber on which reside the segment
  virtual DTChamberId chamberId() const;
    
 private:
  
  /// the superPhi segment 
  DTChamberRecSegment2D *phiSegment() {return &thePhiSeg;}
    
  /// the Z segment
  DTSLRecSegment2D *zSegment() {return &theZedSeg;}

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

  DTChamberRecSegment2D thePhiSeg;
  DTSLRecSegment2D theZedSeg;

  AlgebraicMatrix theProjMatrix;  // the projection matrix
  int theDimension; // the dimension of this rechit

  DetId theDetId;           // Id of the det this seg belongs

};

std::ostream& operator<<(std::ostream& os, const DTRecSegment4D& seg);

#endif // DTRecHit_DTRecSegment4D_h

