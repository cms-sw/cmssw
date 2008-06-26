#ifndef confidence_h
#define confidence_h

#include <utility>

/* 
Confidence level estimators for a Binomial distribution.

For a detailed discussion see:
    "Interval Estimation for a Binomial Proportion", 
    L. D. Brown, T. T. Cai and A. DasGupta, 
    2001, Statistical Sciences 16 (2) 101-133, 
    http://www-stat.wharton.upenn.edu/~tcai/paper/Binomial-StatSci.pdf

The D0 implementation is detailed in the source code and documentation for the class TGraphAsymmErrors.

Implementation by Andrea Bocci <andrea.bocci@cern.ch>
Last modified on Wed Jun 25 19:23:52 CEST 2008

This is free software, licenced under the GNU LGPL version 2.1, or (at your option) any later version.
*/

namespace confidence {

typedef std::pair<double,double> interval;

// default implementation
interval confidence_binomial(unsigned int n, unsigned int k, double level);

// Clopper-Pearson "exact" interval
interval confidence_binomial_clopper_pearson(unsigned int n, unsigned int k, double level);

// normal approximation
interval confidence_binomial_normal(unsigned int n, unsigned int k, double level);

// Jeffreys prior interval
interval confidence_binomial_jeffreys(unsigned int n, unsigned int k, double level);

// modified Jeffreys prior interval
interval confidence_binomial_jeffreys_modified(unsigned int n, unsigned int k, double level);

// alternative bayesian approach, as implemented by the D0 collaboration
interval confidence_binomial_d0(unsigned int n, unsigned int k, double level);

} // namespace confidence

#endif // confidence_h
