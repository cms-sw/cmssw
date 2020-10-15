#ifndef DataFormats_TrackerRecHit2D_VectorHit2D_h
#define DataFormats_TrackerRecHit2D_VectorHit2D_h

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"

//Stores geometric information about VectorHits, their convariance matrix and chi2 of the compatability of the two hits

class VectorHit2D {
public:
  VectorHit2D() : thePosition(), theDirection(), theCovMatrix(), theChi2() {}
  VectorHit2D(const LocalPoint& pos, const LocalVector& dir, const AlgebraicSymMatrix22& covMatrix, const double& chi2)
      : thePosition(pos),
        theDirection(dir),
        theCovMatrix(covMatrix),
        theLocalError(theCovMatrix[0][0], theCovMatrix[0][1], theCovMatrix[1][1]),
        theChi2(chi2) {}

  const LocalPoint& localPosition() const { return thePosition; }
  const LocalVector& localDirection() const { return theDirection; }
  const LocalError& localDirectionError() const { return theLocalError; }
  const AlgebraicSymMatrix22& covMatrix() const { return theCovMatrix; }
  float chi2() const { return theChi2; }
  int dimension() const { return theDimension; }

private:
  LocalPoint thePosition;
  LocalVector theDirection;
  AlgebraicSymMatrix22 theCovMatrix;
  LocalError theLocalError;
  float theChi2;
  static constexpr int theDimension = 2;
};
#endif
