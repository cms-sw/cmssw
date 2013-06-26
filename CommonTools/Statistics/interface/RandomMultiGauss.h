#ifndef _CommonDet_RandomMultiGauss_H_
#define _CommonDet_RandomMultiGauss_H_

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/** \class RandomMultiGauss
 * Vector of random numbers according to covariance matrix.
 * Generates vectors of random numbers given a vector of
 * mean values (optional) and a covariance matrix.
 * Will accept empty rows/columns in the input matrix.
 * Uses CLHEP::RandGauss with default engine for generation.
 */

class RandomMultiGauss {
public:
  /** constructor with explicit vector of mean values
   */
  RandomMultiGauss (const AlgebraicVector& aVector, const AlgebraicSymMatrix& aMatrix);
  /** constructor with covariance matrix only (all means = 0)
   */
  RandomMultiGauss (const AlgebraicSymMatrix& aMatrix);
  // destructor
  ~RandomMultiGauss() {}
  /** Generation of a vector of random numbers.
   */
  AlgebraicVector fire();
  
private:
  /** generation of the cholesky decomposition
   */
  void initialise(const AlgebraicSymMatrix&);

private:
  int theSize;
  AlgebraicVector theMeans;
  AlgebraicMatrix theTriangle;
};

#endif
