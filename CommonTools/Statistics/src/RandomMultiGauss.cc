//#include "CommonDet/DetUtilities/interface/DetExceptions.h"
#include "CommonTools/Statistics/interface/RandomMultiGauss.h"
#include "CLHEP/Random/RandGauss.h"

#include <cfloat>
//
// constructor with means and covariance
//
RandomMultiGauss::RandomMultiGauss (const AlgebraicVector& aVector, const AlgebraicSymMatrix& aMatrix) :
  theSize(aMatrix.num_row()),
  theMeans(aVector),
  theTriangle(theSize,theSize,0) {
  //
  // Check consistency
  //
  if ( theMeans.num_row() == theSize ) {
    initialise(aMatrix);
  }
  else {
//    throw DetLogicError("RandomMultiGauss: size of vector and matrix do not match");
    theMeans = AlgebraicVector(theSize,0);
  }
}
//
// constructor with covariance (mean = 0)
//
RandomMultiGauss::RandomMultiGauss (const AlgebraicSymMatrix& aMatrix) :
  theSize(aMatrix.num_row()),
  theMeans(theSize,0),
  theTriangle(theSize,theSize,0) {
  //
  initialise(aMatrix);
}
//
// construct triangular matrix (Cholesky decomposition)
//
void RandomMultiGauss::initialise (const AlgebraicSymMatrix& aMatrix) {
  //
  // Cholesky decomposition with protection against empty rows/columns
  //
  for ( int i1=0; i1<theSize; i1++ ) {
    if ( fabs(aMatrix[i1][i1])<FLT_MIN )  continue;

    for ( int i2=i1; i2<theSize; i2++ ) {
      if ( fabs(aMatrix[i2][i2])<FLT_MIN )  continue;

      double sum = aMatrix[i2][i1];
      for ( int i3=i1-1; i3>=0; i3-- ) {
	if ( fabs(aMatrix[i3][i3])<FLT_MIN )  continue;
	sum -= theTriangle[i1][i3]*theTriangle[i2][i3];
      }
      
      if ( i1==i2 ) {
	//
	// check for positive definite input matrix, but allow for effects
	// due to finite precision
	//
	if ( sum<=0 ) {
//	  if ( sum<-FLT_MIN )  throw DetLogicError("RandomMultiGauss: input matrix is not positive definite");
	  sum = FLT_MIN;
	}
	theTriangle[i1][i1] = sqrt(sum);
      }
      else {
	theTriangle[i1][i2] = 0.;
	theTriangle[i2][i1] = sum / theTriangle[i1][i1];
      }
    }
  }
}
//
// generate vector of random numbers
//
AlgebraicVector RandomMultiGauss::fire() {
  AlgebraicVector vRandom(theSize,0);
  for ( int i=0; i<theSize; i++ ) {
    if ( theTriangle[i][i]!=0 ) 
      vRandom[i] = CLHEP::RandGauss::shoot();
  }
  return theTriangle*vRandom+theMeans;
}

