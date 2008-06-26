#include <limits>
#include <cmath>

#include "Math/QuantFuncMathCore.h"
#include "TGraphAsymmErrors.h"

#include "confidence.h"

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

// default implementation
interval confidence_binomial(unsigned int n, unsigned int k, double level) {
  return confidence_binomial_jeffreys(n, k, level);
}


// Clopper-Pearson "exact" interval
interval confidence_binomial_clopper_pearson(unsigned int n, unsigned int k, double level) {
  double min = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();

  if (k <= n) {
    double alpha = (1.0 - level) / 2;
    min = (k == 0) ? 0.0 : ROOT::Math::beta_quantile(      alpha, k,       n-k + 1.0);
    max = (k == n) ? 1.0 : ROOT::Math::beta_quantile(1.0 - alpha, k + 1.0, n-k);
  }
  return std::make_pair(min, max);
}

//#include <iostream>
// normal approximation
interval confidence_binomial_normal(unsigned int n, unsigned int k, double level) {
  double min = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();

  if (k <= n) {
    double alpha = (1.0 - level) / 2;
    double average = (double) k / n;
    double sigma = std::sqrt(average * (1-average) / n);
    double delta = ROOT::Math::normal_quantile(1.0 - alpha, sigma);
    min = ((average - delta) < 0.) ? 0. : (average - delta);
    max = ((average + delta) > 1.) ? 1. : (average + delta);
  }
  return std::make_pair(min, max);
}

// Jeffreys prior interval
interval confidence_binomial_jeffreys(unsigned int n, unsigned int k, double level) {
  double min = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();

  if (k <= n) {
    double alpha = (1.0 - level) / 2;
    min = (k == 0) ? 0.0 : ROOT::Math::beta_quantile(      alpha, k + 0.5, n-k + 0.5);
    max = (k == n) ? 1.0 : ROOT::Math::beta_quantile(1.0 - alpha, k + 0.5, n-k + 0.5);
  }
  return std::make_pair(min, max);
}

// modified Jeffreys prior interval
interval confidence_binomial_jeffreys_modified(unsigned int n, unsigned int k, double level) {
  double min = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();

  if (k <= n) {
    double alpha = (1.0 - level) / 2;
    double correction = 1 - std::pow(alpha, 1./n);
    if (n == 0) {
      // (0, 0)
      min = 0.0;
      max = 1.0;
    } else if (k == 0) {
      // (N, 0)
      min = 0.0;
      max = correction;
    } else if (k == n) {
      // (N, N)
      min = 1.0 - correction;
      max = 1.0;
    } else {
      min = (k ==   1) ? 0.0 : ROOT::Math::beta_quantile(      alpha, k + 0.5, n-k + 0.5);
      max = (k == n-1) ? 1.0 : ROOT::Math::beta_quantile(1.0 - alpha, k + 0.5, n-k + 0.5);
    }
  }
  return std::make_pair(min, max);
}

interval confidence_binomial_d0(unsigned int n, unsigned int k, double level) {
  // hack to access the implementation used in TGraphAsymmErrors (develped by D0) 
  struct AccessTGAEEfficicency : private TGraphAsymmErrors {
    void confidence_binomial(unsigned int n, unsigned int k, double level, double & min, double & max) {
      double mode;
      TGraphAsymmErrors::Efficiency(k, n, level, mode, min, max);
    }
  };

  static AccessTGAEEfficicency d0;
  double min = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();

  if (k <= n) {
    d0.confidence_binomial(n, k, level, min, max);
  }
  return std::make_pair(min, max);
}

} // namespace confidence
