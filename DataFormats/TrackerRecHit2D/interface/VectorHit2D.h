#ifndef TrackerRecHit2D_VectorHit2D_h
#define TrackerRecHit2D_VectorHit2D_h

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"

class VectorHit2D {
public:
  VectorHit2D() : thePosition(), theDirection(), theCovMatrix(), theChi2(), theDimension(2) {}
  VectorHit2D(const LocalPoint& pos, const LocalVector& dir, const AlgebraicSymMatrix22& covMatrix, const double& Chi2)
      : thePosition(pos), theDirection(dir), theCovMatrix(covMatrix), theChi2(Chi2), theDimension(2){};
  virtual ~VectorHit2D(){};

  LocalPoint localPosition() const { return thePosition; }
  LocalVector localDirection() const { return theDirection; }
  LocalError localDirectionError() const {
    return LocalError(theCovMatrix[0][0], theCovMatrix[0][1], theCovMatrix[1][1]);
  }
  AlgebraicSymMatrix22 covMatrix() const { return theCovMatrix; }
  double chi2() const { return theChi2; }
  int dimension() const { return 2; }

private:
  LocalPoint thePosition;
  LocalVector theDirection;
  AlgebraicSymMatrix22 theCovMatrix;
  double theChi2;
  int theDimension;
};
#endif
