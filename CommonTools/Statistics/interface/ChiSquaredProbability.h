#ifndef ChiSquaredProbability_H
#define ChiSquaredProbability_H

/** Returns the probability that an observation, correctly described by
 *  a model with nrDOF, will give rise to a chi-squared larger than the one
 *  observed; from this, one can interpret this probability as how likely
 *  it is to observe as high (or higher) a chi-squared. 
 *  source: Numerical Recipes
 */
float ChiSquaredProbability( double chiSquared, double nrDOF );
float LnChiSquaredProbability( double chiSquared, double nrDOF );

#endif
