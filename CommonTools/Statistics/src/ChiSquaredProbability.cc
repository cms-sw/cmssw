#ifndef ChiSquaredProbability_H
#define ChiSquaredProbability_H

#include "CommonTools/Statistics/src/IncompleteGammaComplement.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

/** Returns the probability that an observation, correctly described by
 *  a model with nrDOF, will give rise to a chi-squared larger than the one
 *  observed; from this, one can interpret this probability as how likely
 *  it is to observe as high (or higher) a chi-squared. 
 *  source: Numerical Recipes
 */
float ChiSquaredProbability( double chiSquared, double nrDOF )
{ return IncompleteGammaComplement::value( nrDOF / 2 , chiSquared / 2 ); }

float LnChiSquaredProbability( double chiSquared, double nrDOF )
{ return IncompleteGammaComplement::ln( nrDOF / 2 , chiSquared / 2 ); }

#endif
