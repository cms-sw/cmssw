#ifndef ChiSquared_H
#define ChiSquared_H


/** \class ChiSquared
 *  Constructed with total chi-squared value `value` and number of degrees 
 *  of freedom `ndf`. <BR>
 *
 *  Computes chi-squared upper tail probability, 
 *  i.e. the probability that an observation, correctly described by
 *  a model with nrDOF, will give rise to a chi-squared larger than the one
 *  observed. From this, one can interpret this probability as how likely
 *  it is to observe as high (or higher) a chi-squared. <BR>
 *
 *  Also computes the natural logarithm of that probability, 
 *  useful to compare very unlikely events, for which the probability 
 *  is rounded off to 0.
 */

class ChiSquared {

public:

  ChiSquared(float value, float ndf) : theValue(value), theNDF(ndf) {}

  float value() const;
  float degreesOfFreedom() const;
  float probability() const;
  float lnProbability() const;

private:

  float theValue, theNDF;

};

#endif
