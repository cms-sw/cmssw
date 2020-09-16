#ifndef GEMRecHit_GEMSegment_h
#define GEMRecHit_GEMSegment_h

/** \class GEMSegment derived by the CSC segment
 *  Describes a reconstructed track segment in the 2+ layers of a GEM chamber.
 *  This is 4-dimensional since it has an origin (x,y) and a direction (x,y)
 *  in the local coordinate system of the chamber.
 *
 *  \author Piet Verwilligen
 */

#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

#include <iosfwd>

class GEMDetId;

class GEMSegment final : public RecSegment {
public:
  /// Default constructor
  GEMSegment() : theChi2(0.) {}

  /// Constructor
  GEMSegment(const std::vector<const GEMRecHit*>& proto_segment,
             const LocalPoint& origin,
             const LocalVector& direction,
             const AlgebraicSymMatrix& errors,
             double chi2);

  GEMSegment(const std::vector<const GEMRecHit*>& proto_segment,
             const LocalPoint& origin,
             const LocalVector& direction,
             const AlgebraicSymMatrix& errors,
             double chi2,
             float bx);

  GEMSegment(const std::vector<const GEMRecHit*>& proto_segment,
             const LocalPoint& origin,
             const LocalVector& direction,
             const AlgebraicSymMatrix& errors,
             double chi2,
             float bx,
             float deltaPhi);

  /// Destructor
  ~GEMSegment() override;

  //--- Base class interface
  GEMSegment* clone() const override { return new GEMSegment(*this); }

  LocalPoint localPosition() const override { return theOrigin; }
  LocalError localPositionError() const override;

  LocalVector localDirection() const override { return theLocalDirection; }
  LocalError localDirectionError() const override;

  /// Parameters of the segment, for the track fit in the order (dx/dz, dy/dz, x, y )
  AlgebraicVector parameters() const override;

  /// Covariance matrix of parameters()
  AlgebraicSymMatrix parametersError() const override { return theCovMatrix; }

  /// The projection matrix relates the trajectory state parameters to the segment parameters().
  AlgebraicMatrix projectionMatrix() const override;

  std::vector<const TrackingRecHit*> recHits() const override;

  std::vector<TrackingRecHit*> recHits() override;

  double chi2() const override { return theChi2; };

  int dimension() const override { return 4; }

  int degreesOfFreedom() const override { return 2 * nRecHits() - 4; }

  //--- Extension of the interface

  const std::vector<GEMRecHit>& specificRecHits() const { return theGEMRecHits; }

  int nRecHits() const { return theGEMRecHits.size(); }

  GEMDetId gemDetId() const { return geographicalId(); }

  float bunchX() const { return theBX; }

  float deltaPhi() const { return theDeltaPhi; }

  void print() const;

private:
  std::vector<GEMRecHit> theGEMRecHits;
  LocalPoint theOrigin;             // in chamber frame - the GeomDet local coordinate system
  LocalVector theLocalDirection;    // in chamber frame - the GeomDet local coordinate system
  AlgebraicSymMatrix theCovMatrix;  // the covariance matrix
  double theChi2;                   // the Chi squared of the segment fit
  float theBX;                      // the bunch crossing
  float theDeltaPhi;                // Difference in segment phi position: outer layer - inner lay
};

std::ostream& operator<<(std::ostream& os, const GEMSegment& seg);

#endif
