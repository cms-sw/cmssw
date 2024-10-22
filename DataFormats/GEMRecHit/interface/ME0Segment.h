#ifndef GEMRecHit_ME0Segment_h
#define GEMRecHit_ME0Segment_h

/** \class ME0Segment derived by the CSC segment
 *  Describes a reconstructed track segment in the 6 layers of the ME0 system.
 *  This is 4-dimensional since it has an origin (x,y) and a direction (x,y)
 *  in the local coordinate system of the chamber.
 *
 *  $Date: 2014/02/04 12:41:32 $
 *  \author Marcello Maggi
 */

#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"

#include <iosfwd>

class ME0DetId;

class ME0Segment final : public RecSegment {
public:
  /// Default constructor
  ME0Segment() : theChi2(0.), theTimeValue(0.), theTimeUncrt(0.), theDeltaPhi(0.) {}

  /// Constructor
  ME0Segment(const std::vector<const ME0RecHit*>& proto_segment,
             const LocalPoint& origin,
             const LocalVector& direction,
             const AlgebraicSymMatrix& errors,
             float chi2);

  ME0Segment(const std::vector<const ME0RecHit*>& proto_segment,
             const LocalPoint& origin,
             const LocalVector& direction,
             const AlgebraicSymMatrix& errors,
             float chi2,
             float time,
             float timeErr,
             float deltaPhi);

  /// Destructor
  ~ME0Segment() override;

  //--- Base class interface
  ME0Segment* clone() const override { return new ME0Segment(*this); }

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

  const std::vector<ME0RecHit>& specificRecHits() const { return theME0RecHits; }

  int nRecHits() const { return theME0RecHits.size(); }

  ME0DetId me0DetId() const { return geographicalId(); }

  float time() const { return theTimeValue; }
  float timeErr() const { return theTimeUncrt; }

  float deltaPhi() const { return theDeltaPhi; }

  void print() const;

private:
  std::vector<ME0RecHit> theME0RecHits;
  LocalPoint theOrigin;             // in chamber frame - the GeomDet local coordinate system
  LocalVector theLocalDirection;    // in chamber frame - the GeomDet local coordinate system
  AlgebraicSymMatrix theCovMatrix;  // the covariance matrix
  float theChi2;                    // the Chi squared of the segment fit
  float theTimeValue;               // the best time estimate of the segment
  float theTimeUncrt;               // the uncertainty on the time estimation
  float theDeltaPhi;                // Difference in segment phi position: outer layer - inner lay
};

std::ostream& operator<<(std::ostream& os, const ME0Segment& seg);

#endif
