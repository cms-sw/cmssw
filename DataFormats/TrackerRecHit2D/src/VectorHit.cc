#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "Geometry/CommonDetUnit/interface/StackGeomDet.h"
#include "CLHEP/Units/PhysicalConstants.h"

VectorHit::VectorHit(const GeomDet& idet,
                     const LocalPoint& posLower,
                     const LocalVector& dir,
                     const AlgebraicSymMatrix44& covMatrix,
                     const float chi2,
                     OmniClusterRef const& lower,
                     OmniClusterRef const& upper,
                     const float curvature,
                     const float curvatureError,
                     const float phi)
    : BaseTrackerRecHit(idet, trackerHitRTTI::vector),
      thePosition(posLower),
      theDirection(dir),
      theCovMatrix(covMatrix),
      theChi2(chi2),
      theLowerCluster(lower),
      theUpperCluster(upper),
      theCurvature(curvature),
      theCurvatureError(curvatureError),
      thePhi(phi) {}

VectorHit::VectorHit(const GeomDet& idet,
                     const VectorHit2D& vh2Dzx,
                     const VectorHit2D& vh2Dzy,
                     OmniClusterRef const& lower,
                     OmniClusterRef const& upper,
                     const float curvature,
                     const float curvatureError,
                     const float phi)
    : BaseTrackerRecHit(idet, trackerHitRTTI::vector),
      thePosition(LocalPoint(vh2Dzx.localPosition().x(), vh2Dzy.localPosition().x(), 0.)),
      theDirection(LocalVector(vh2Dzx.localDirection().x(), vh2Dzy.localDirection().x(), 1.)),
      theLowerCluster(lower),
      theUpperCluster(upper),
      theCurvature(curvature),
      theCurvatureError(curvatureError),
      thePhi(phi) {
  //building the cov matrix 4x4 starting from the 2x2
  const AlgebraicSymMatrix22& covMatZX = vh2Dzx.covMatrix();
  const AlgebraicSymMatrix22& covMatZY = vh2Dzy.covMatrix();

  theCovMatrix = AlgebraicSymMatrix44();

  theCovMatrix[0][0] = covMatZX[0][0];  // var(dx/dz)
  theCovMatrix[1][1] = covMatZY[0][0];  // var(dy/dz)
  theCovMatrix[2][2] = covMatZX[1][1];  // var(x)
  theCovMatrix[3][3] = covMatZY[1][1];  // var(y)
  theCovMatrix[0][2] = covMatZX[0][1];  // cov(dx/dz,x)
  theCovMatrix[1][3] = covMatZY[0][1];  // cov(dy/dz,y)

  theChi2 = vh2Dzx.chi2() + vh2Dzy.chi2();
}

bool VectorHit::sharesInput(const TrackingRecHit* other, SharedInputType what) const {
  if (what == all && (geographicalId() != other->geographicalId()))
    return false;

  if (!sameDetModule(*other))
    return false;

  if (trackerHitRTTI::isVector(*other)) {
    const VectorHit* otherVh = static_cast<const VectorHit*>(other);
    return sharesClusters(*otherVh, what);
  }

  if (what == all)
    return false;

  // what about multi???
  auto const& otherClus = reinterpret_cast<const BaseTrackerRecHit*>(other)->firstClusterRef();
  return (otherClus == lowerClusterRef()) || (otherClus == upperClusterRef());
}

bool VectorHit::sharesClusters(VectorHit const& other, SharedInputType what) const {
  bool lower = this->lowerClusterRef() == other.lowerClusterRef();
  bool upper = this->upperClusterRef() == other.upperClusterRef();

  return (what == TrackingRecHit::all) ? (lower && upper) : (upper || lower);
}

void VectorHit::getKfComponents4D(KfComponentsHolder& holder) const {
  AlgebraicVector4& pars = holder.params<theDimension>();
  pars[0] = theDirection.x();
  pars[1] = theDirection.y();
  pars[2] = thePosition.x();
  pars[3] = thePosition.y();

  holder.errors<theDimension>() = theCovMatrix;

  ProjectMatrix<double, 5, theDimension>& pf = holder.projFunc<theDimension>();
  for (int i = 0; i < 4; ++i)
    pf.index[i] = i + 1;

  holder.measuredParams<theDimension>() = AlgebraicVector4(&holder.tsosLocalParameters().At(1), theDimension);
  holder.measuredErrors<theDimension>() = holder.tsosLocalErrors().Sub<AlgebraicSymMatrix44>(1, 1);
}

Global3DPoint VectorHit::lowerGlobalPos() const {
  const StackGeomDet* stackDet = static_cast<const StackGeomDet*>(det());
  const PixelGeomDetUnit* geomDetLower = static_cast<const PixelGeomDetUnit*>(stackDet->lowerDet());
  return phase2clusterGlobalPos(geomDetLower, lowerCluster());
}

Global3DPoint VectorHit::upperGlobalPos() const {
  const StackGeomDet* stackDet = static_cast<const StackGeomDet*>(det());
  const PixelGeomDetUnit* geomDetUpper = static_cast<const PixelGeomDetUnit*>(stackDet->upperDet());
  return phase2clusterGlobalPos(geomDetUpper, upperCluster());
}

Global3DPoint VectorHit::phase2clusterGlobalPos(const PixelGeomDetUnit* geomDet, ClusterRef cluster) {
  const PixelTopology* topo = &geomDet->specificTopology();
  float ix = cluster->center();
  float iy = cluster->column() + 0.5f;                   // halfway the column
  LocalPoint lp(topo->localX(ix), topo->localY(iy), 0);  // x, y, z
  Global3DPoint gp = geomDet->surface().toGlobal(lp);
  return gp;
}

GlobalError VectorHit::lowerGlobalPosErr() const {
  const StackGeomDet* stackDet = static_cast<const StackGeomDet*>(det());
  const PixelGeomDetUnit* geomDetLower = static_cast<const PixelGeomDetUnit*>(stackDet->lowerDet());
  return phase2clusterGlobalPosErr(geomDetLower);
}

GlobalError VectorHit::upperGlobalPosErr() const {
  const StackGeomDet* stackDet = static_cast<const StackGeomDet*>(det());
  const PixelGeomDetUnit* geomDetUpper = static_cast<const PixelGeomDetUnit*>(stackDet->upperDet());
  return phase2clusterGlobalPosErr(geomDetUpper);
}

GlobalError VectorHit::phase2clusterGlobalPosErr(const PixelGeomDetUnit* geomDet) {
  const PixelTopology* topo = &geomDet->specificTopology();
  float pitchX = topo->pitch().first;
  float pitchY = topo->pitch().second;
  constexpr float invTwelve = 1. / 12;
  LocalError le(pow(pitchX, 2) * invTwelve, 0, pow(pitchY, 2) * invTwelve);  // e2_xx, e2_xy, e2_yy
  GlobalError ge(ErrorFrameTransformer().transform(le, geomDet->surface()));
  return ge;
}

Global3DVector VectorHit::globalDirectionVH() const {
  Local3DVector theLocalDelta =
      LocalVector(theDirection.x() * theDirection.z(), theDirection.y() * theDirection.z(), theDirection.z());
  Global3DVector g = det()->surface().toGlobal(theLocalDelta);
  return g;
}

Global3DVector VectorHit::globalDirection() const { return (det()->surface().toGlobal(localDirection())); }

float VectorHit::theta() const { return globalDirection().theta(); }

float VectorHit::transverseMomentum(float magField) const {
  return magField * (CLHEP::c_light * 1e-5F) / theCurvature;
  // pT [GeV] ~ 0.3 * B[T] * R [m], curvature is in cms, using precise value from speed of light
  // because curvature is a signed quantity this transverse momentum is also signed
}
float VectorHit::momentum(float magField) const { return transverseMomentum(magField) / (1. * sin(theta())); }

LocalError VectorHit::localPositionError() const {
  return LocalError(theCovMatrix[2][2], theCovMatrix[2][3], theCovMatrix[3][3]);
}

LocalError VectorHit::localDirectionError() const {
  return LocalError(theCovMatrix[0][0], theCovMatrix[0][1], theCovMatrix[1][1]);
}

const AlgebraicSymMatrix44& VectorHit::covMatrix() const { return theCovMatrix; }

std::ostream& operator<<(std::ostream& os, const VectorHit& vh) {
  os << " VectorHit create in the DetId#: " << vh.geographicalId() << "\n"
     << " Vectorhit local position      : " << vh.localPosition() << "\n"
     << " Vectorhit local direction     : " << vh.localDirection() << "\n"
     << " Vectorhit global direction    : " << vh.globalDirection() << "\n"
     << " Lower cluster global position : " << vh.lowerGlobalPos() << "\n"
     << " Upper cluster global position : " << vh.upperGlobalPos();

  return os;
}

/// Access to component RecHits (if any)
std::vector<const TrackingRecHit*> VectorHit::recHits() const { return {}; }

/// Non-const access to component RecHits (if any)
std::vector<TrackingRecHit*> VectorHit::recHits() { return {}; }
