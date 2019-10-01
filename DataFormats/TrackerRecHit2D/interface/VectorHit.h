#ifndef TrackerRecHit2D_VectorHit_h
#define TrackerRecHit2D_VectorHit_h

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
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"

#include "TkCloner.h"

class VectorHit GCC11_FINAL : public BaseTrackerRecHit {

  public:

  typedef OmniClusterRef::Phase2Cluster1DRef ClusterRef;

  VectorHit() : thePosition(), theDirection(), theCovMatrix(), theDimension(0) { setType(bad); }

  VectorHit(const VectorHit& vh) ;

  VectorHit(const GeomDet& idet, const LocalPoint& posInner, const LocalVector& dir,
            const AlgebraicSymMatrix& covMatrix, const double& Chi2,
            OmniClusterRef const& lower, OmniClusterRef const& upper) ;

  VectorHit(const GeomDet& idet, const VectorHit2D& vh2Dzx, const VectorHit2D& vh2Dzy,
            OmniClusterRef const& lower, OmniClusterRef const& upper) ;

  ~VectorHit() ;

  virtual VectorHit* clone() const override { return new VectorHit(*this);}
#ifndef __GCCXML__
  virtual RecHitPointer cloneSH() const override { return std::make_shared<VectorHit>(*this);}
#endif

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const override;
  bool sharesClusters(VectorHit const & h1, VectorHit const & h2,
                      SharedInputType what) const ;

  // Parameters of the segment, for the track fit
  // For a 4D segment: (dx/dz,dy/dz,x,y)
  bool hasPositionAndError() const  GCC11_FINAL{
  //bool hasPositionAndError() const {
    return true;
//      return (err_.xx() != 0) || (err_.yy() != 0) || (err_.xy() != 0) ||
//             (pos_.x()  != 0) || (pos_.y()  != 0) || (pos_.z()  != 0);
  };

  virtual AlgebraicVector parameters() const override;
  virtual void getKfComponents( KfComponentsHolder & holder ) const override { getKfComponents4D(holder); }
  void getKfComponents4D( KfComponentsHolder & holder ) const ;

  // returning methods
  LocalPoint localPosition() const GCC11_FINAL { return thePosition; }
  virtual LocalVector localDirection() const { return theDirection; }
  AlgebraicSymMatrix parametersError() const override ;
  LocalError localPositionError() const GCC11_FINAL ;
  virtual LocalError localDirectionError() const ;
  Global3DVector globalDirection() const;

  virtual double chi2() const { return theChi2; }
  virtual int dimension() const override { return theDimension; }

  std::pair<double,double> curvatureORphi(std::string curvORphi = "curvature") const ;
  float transverseMomentum(const MagneticField* magField);
  float momentum(const MagneticField* magField);

  ClusterRef lowerCluster() const { return theLowerCluster.cluster_phase2OT(); }
  ClusterRef upperCluster() const { return theUpperCluster.cluster_phase2OT(); }
  OmniClusterRef const lowerClusterRef() const { return theLowerCluster; }
  OmniClusterRef const upperClusterRef() const { return theUpperCluster; }

  //FIXME::to update with a proper CPE maybe...
  Global3DPoint lowerGlobalPos() const ;
  Global3DPoint upperGlobalPos() const ;
  Global3DPoint phase2clusterGlobalPos(const PixelGeomDetUnit* geomDet, ClusterRef cluster) const;
  GlobalError lowerGlobalPosErr() const ;
  GlobalError upperGlobalPosErr() const ;
  GlobalError phase2clusterGlobalPosErr(const PixelGeomDetUnit* geomDet) const;

  virtual bool isPhase2() const override { return true; }

  //FIXME: I have always two clusters in a VH
  virtual OmniClusterRef const & firstClusterRef() const GCC11_FINAL { return theLowerCluster;}
  ClusterRef cluster()  const { return theLowerCluster.cluster_phase2OT(); }

  //This method returns the delta in global coordinates
  Global3DVector globalDelta() const;
  float theta();

  /// The projection matrix relates the trajectory state parameters to the segment parameters().
  virtual AlgebraicMatrix projectionMatrix() const override;

  // Access to component RecHits (if any)
  virtual std::vector<const TrackingRecHit*> recHits() const override;
  virtual std::vector<TrackingRecHit*> recHits() override ;

  // setting methods
  void setPosition(LocalPoint pos) { thePosition = pos; }
  void setDirection(LocalVector dir) { theDirection = dir; }
  void setCovMatrix(AlgebraicSymMatrix mat) { theCovMatrix = mat; }

 private:
  // double dispatch
  virtual VectorHit * clone_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override{
    return cloner(*this,tsos).release();
  }
  virtual  RecHitPointer cloneSH_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override{
    return cloner.makeShared(*this,tsos);
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
  AlgebraicSymMatrix theCovMatrix;
  double theChi2;
  int theDimension;
  OmniClusterRef theLowerCluster;
  OmniClusterRef theUpperCluster;

};

inline bool operator<( const VectorHit& one, const VectorHit& other) {

  if ( one.chi2() < other.chi2() ) {
    return true;
  }

  return false;
}

std::ostream& operator<<(std::ostream& os, const VectorHit& vh);

typedef edmNew::DetSetVector<VectorHit> VectorHitCollection;
typedef VectorHitCollection             VectorHitCollectionNew;

#endif
