#ifndef DTRecHit_DTRecSegment4D_h
#define DTRecHit_DTRecSegment4D_h

/** \class DTRecSegment4D
 *
 * 4-parameter RecHits for MuonBarrel DT (x,y, dx/dz, dy/dz)
 *
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"

/* Collaborating Class Declarations */
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"

/* C++ Headers */
#include <iosfwd>

class DTRecSegment4D : public RecSegment {
public:
  friend class DTSegmentUpdator;
  /// Empty constructor
  DTRecSegment4D() : theProjection(none), theDimension(0) {}

  /// Construct from phi and Z projections
  DTRecSegment4D(const DTChamberRecSegment2D& phiSeg,
                 const DTSLRecSegment2D& zedSeg,
                 const LocalPoint& posZInCh,
                 const LocalVector& dirZInCh);

  /// Construct from phi projection
  DTRecSegment4D(const DTChamberRecSegment2D& phiSeg);

  /// Construct from Z projection
  DTRecSegment4D(const DTSLRecSegment2D& zedSeg, const LocalPoint& posZInCh, const LocalVector& dirZInCh);

  /// Destructor
  ~DTRecSegment4D() override;

  //--- Base class interface

  DTRecSegment4D* clone() const override { return new DTRecSegment4D(*this); }

  /// Parameters of the segment, for the track fit.
  /// For a 4D segment: (dx/dy,dy/dz,x,y)
  /// For a 2D, phi-only segment: (dx/dz,x)
  /// For a 2D, Z-only segment: (dy/dz,y)
  AlgebraicVector parameters() const override;

  /// Covariance matrix fo parameters()
  AlgebraicSymMatrix parametersError() const override;

  /// The projection matrix relates the trajectory state parameters to the segment parameters().
  AlgebraicMatrix projectionMatrix() const override;

  /// Local position in Chamber frame
  LocalPoint localPosition() const override { return thePosition; }

  /// Local position error in Chamber frame
  LocalError localPositionError() const override;

  /// Local direction in Chamber frame
  LocalVector localDirection() const override { return theDirection; }

  /// Local direction error in the Chamber frame
  LocalError localDirectionError() const override;

  // Chi2 of the segment fit
  double chi2() const override;

  // Degrees of freedom of the segment fit
  int degreesOfFreedom() const override;

  // Dimension (in parameter space)
  int dimension() const override { return theDimension; }

  // Access to component RecHits (if any)
  std::vector<const TrackingRecHit*> recHits() const override;

  // Non-const access to component RecHits (if any)
  std::vector<TrackingRecHit*> recHits() override;

  //--- Extension of the interface

  /// Does it have the Phi projection?
  bool hasPhi() const { return (theProjection == full || theProjection == phi); }

  /// Does it have the Z projection?
  bool hasZed() const { return (theProjection == full || theProjection == Z); }

  /// The superPhi segment: 0 if no phi projection available
  const DTChamberRecSegment2D* phiSegment() const { return hasPhi() ? &thePhiSeg : nullptr; }

  /// The Z segment: 0 if not zed projection available
  const DTSLRecSegment2D* zSegment() const { return hasZed() ? &theZedSeg : nullptr; }

  /// Set position
  void setPosition(LocalPoint pos) { thePosition = pos; }

  /// Set direction
  void setDirection(LocalVector dir) { theDirection = dir; }

  /// Set covariance matrix
  void setCovMatrix(const AlgebraicSymMatrix& mat) { theCovMatrix = mat; }

  /// The (specific) DetId of the chamber on which the segment resides
  virtual DTChamberId chamberId() const;

private:
  /// Which projections are actually there
  enum Projection { full, phi, Z, none };
  Projection theProjection;

  /// the superPhi segment
  DTChamberRecSegment2D* phiSegment() { return &thePhiSeg; }

  /// the Z segment
  DTSLRecSegment2D* zSegment() { return &theZedSeg; }

  LocalPoint thePosition;    // in chamber frame
  LocalVector theDirection;  // in chamber frame

  void setCovMatrixForZed(const LocalPoint& posZInCh);

  // the covariance matrix, has the following meaning
  // mat[0][0]=sigma (dx/dz)
  // mat[1][1]=sigma (dy/dz)
  // mat[2][2]=sigma (x)
  // mat[3][3]=sigma (y)
  // mat[0][2]=cov(dx/dz,x)
  // mat[1][3]=cov(dy/dz,y)
  AlgebraicSymMatrix theCovMatrix;

  DTChamberRecSegment2D thePhiSeg;
  DTSLRecSegment2D theZedSeg;

  int theDimension;  // the dimension of this rechit
};

std::ostream& operator<<(std::ostream& os, const DTRecSegment4D& seg);

#endif  // DTRecHit_DTRecSegment4D_h
