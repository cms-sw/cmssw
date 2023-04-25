#ifndef DataFormats_TrackerRecHit2D_VectorHit_h
#define DataFormats_TrackerRecHit2D_VectorHit_h

/** \class VectorHit
 *
 * 4-parameter RecHits for Phase2 Tracker (x,y, dx/dz, dy/dz)
 *
 * $Date: 2015/03/30 $
 * \author Erica Brondolin
 *
 */

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"

#include "DataFormats/TrackerRecHit2D/interface/TkCloner.h"

class VectorHit final : public BaseTrackerRecHit {
public:
  typedef OmniClusterRef::Phase2Cluster1DRef ClusterRef;

  VectorHit() : thePosition(), theDirection(), theCovMatrix() { setType(bad); }

  VectorHit(const GeomDet& idet,
            const LocalPoint& posInner,
            const LocalVector& dir,
            const AlgebraicSymMatrix44& covMatrix,
            const float chi2,
            OmniClusterRef const& lower,
            OmniClusterRef const& upper,
            const float curvature,
            const float curvatureError,
            const float phi);

  VectorHit(const GeomDet& idet,
            const VectorHit2D& vh2Dzx,
            const VectorHit2D& vh2Dzy,
            OmniClusterRef const& lower,
            OmniClusterRef const& upper,
            const float curvature,
            const float curvatureError,
            const float phi);

  ~VectorHit() override = default;

  VectorHit* clone() const override { return new VectorHit(*this); }
  RecHitPointer cloneSH() const override { return std::make_shared<VectorHit>(*this); }

  bool sharesInput(const TrackingRecHit* other, SharedInputType what) const override;
  bool sharesClusters(VectorHit const& other, SharedInputType what) const;

  // Parameters of the segment, for the track fit
  // For a 4D segment: (dx/dz,dy/dz,x,y)
  bool hasPositionAndError() const override {
    //if det is present pos&err are available as well.
    //if det() is not present (null) the hit has been read from file and not updated
    return det();
  };

  void getKfComponents(KfComponentsHolder& holder) const override { getKfComponents4D(holder); }
  void getKfComponents4D(KfComponentsHolder& holder) const;

  // returning methods
  LocalPoint localPosition() const override { return thePosition; }
  virtual LocalVector localDirection() const { return theDirection; }
  const AlgebraicSymMatrix44& covMatrix() const;
  LocalError localPositionError() const override;
  LocalError localDirectionError() const;
  Global3DVector globalDirectionVH() const;

  float chi2() const { return theChi2; }
  int dimension() const override { return theDimension; }
  float curvature() const { return theCurvature; }
  float curvatureError() const { return theCurvatureError; }
  float phi() const { return thePhi; }

  float transverseMomentum(float magField) const;
  float momentum(float magField) const;

  /// "lower" is logical, not geometrically lower; in pixel-strip modules the "lower" is always a pixel
  ClusterRef lowerCluster() const { return theLowerCluster.cluster_phase2OT(); }
  ClusterRef upperCluster() const { return theUpperCluster.cluster_phase2OT(); }
  OmniClusterRef const lowerClusterRef() const { return theLowerCluster; }
  OmniClusterRef const upperClusterRef() const { return theUpperCluster; }
  // Non const variants needed for cluster re-keying
  OmniClusterRef& lowerClusterRef() { return theLowerCluster; }
  OmniClusterRef& upperClusterRef() { return theUpperCluster; }

  //FIXME::to update with a proper CPE maybe...
  Global3DPoint lowerGlobalPos() const;
  Global3DPoint upperGlobalPos() const;
  static Global3DPoint phase2clusterGlobalPos(const PixelGeomDetUnit* geomDet, ClusterRef cluster);
  GlobalError lowerGlobalPosErr() const;
  GlobalError upperGlobalPosErr() const;
  static GlobalError phase2clusterGlobalPosErr(const PixelGeomDetUnit* geomDet);

  bool isPhase2() const override { return true; }

  //FIXME: I have always two clusters in a VH
  OmniClusterRef const& firstClusterRef() const override { return theLowerCluster; }
  ClusterRef cluster() const { return theLowerCluster.cluster_phase2OT(); }

  //This method returns the direction of the segment/stub in global coordinates
  Global3DVector globalDirection() const;
  float theta() const;

  // Access to component RecHits (if any)
  std::vector<const TrackingRecHit*> recHits() const override;
  std::vector<TrackingRecHit*> recHits() override;

private:
  // double dispatch
  VectorHit* clone_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner(*this, tsos).release();
  }
  RecHitPointer cloneSH_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner.makeShared(*this, tsos);
  }

  LocalPoint thePosition;
  LocalVector theDirection;

  // the covariance matrix, has the following meaning
  // mat[0][0]=var(dx/dz)
  // mat[1][1]=var(dy/dz)
  // mat[2][2]=var(x)
  // mat[3][3]=var(y)
  // mat[0][2]=cov(dx/dz,x)
  // mat[1][3]=cov(dy/dz,y)
  AlgebraicSymMatrix44 theCovMatrix;
  float theChi2;
  static constexpr int theDimension = 4;
  OmniClusterRef theLowerCluster;
  OmniClusterRef theUpperCluster;
  float theCurvature;
  float theCurvatureError;
  float thePhi;
};

inline bool operator<(const VectorHit& one, const VectorHit& other) { return (one.chi2() < other.chi2()); }

std::ostream& operator<<(std::ostream& os, const VectorHit& vh);

typedef edmNew::DetSetVector<VectorHit> VectorHitCollection;

#endif
