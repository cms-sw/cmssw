//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprDataMoments.hh,v 1.3 2006/11/13 19:09:39 narsky Exp $
//
// Description:
//      Class SprDataMoments :
//         Computes data means, variances, correlations and kurtosis.
//         Checks hypothesis of zero correlation between two variables.
/*
  The zero correlation hypothesis is tested using the fact that the quantity:

  sqrt(N/(K+1))*(rij-rhoij)/sqrt(1-rij^2)

  converges in distribution to N(0,1) (normal with zero mean and unit
  variance) for elliptical distributions (that is, densities that can
  be described by xAx = const, where A is an inverse of the covariance
  matrix). In formula above, N is the size sample, K is kurtosis, rij
  is an estimator of the correlation between variates i and j, and
  rhoij is the true correlation between these variates.

  For more details on multivariate analysis, see, e.g.,
    Anderson "An Introduction to Multivariate Statistical Analysis"

    Note that all covariances are estimated using the max likelihood
    estimator 1/N sum_ij (x_i-mean_i)*(x_j-mean_j) as opposed to the
    unbiased estimator 1/(N-1) sum_ij (x_i-mean_i)*(x_j-mean_j).
*/
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprDataMoments_HH
#define _SprDataMoments_HH

#include "PhysicsTools/StatPatternRecognition/src/SprSymMatrix.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprVector.hh"

class SprAbsFilter;


class SprDataMoments
{
public:
  virtual ~SprDataMoments() {}

  SprDataMoments(const SprAbsFilter* data) : data_(data) {}

  // moments by index
  double mean(int i) const;
  double variance(int i, double& mean) const;
  double correl(int i, int j, double& mean1, double& mean2, 
		double& var1, double& var2) const;
  double correl(int i, int j) const {
    double mean1(0), mean2(0), var1(0), var2(0);
    return this->correl(i,j,mean1,mean2,var1,var2);
  }
  double kurtosis(SprSymMatrix& cov, SprVector& mean) const;

  // covariance matrix
  bool covariance(SprSymMatrix& cov, SprVector& mean) const;

  // correlation with the class label
  double correlClassLabel(int d, double& mean, double& var) const;

  // test of zero correlation between two variables
  double zeroCorrCL(double corrij, double kurtosis) const;
  double zeroCorrCL(int i, int j) const;

  // moments by name
  double mean(const char* name) const;
  double variance(const char* name, double& mean) const;
  double correl(const char* name1, const char* name2, 
		double& mean1, double& mean2, 
		double& var1, double& var2) const;

  // mean and correlations for the absolute value of a given variable
  double absMean(int i) const;
  double absCorrelClassLabel(int d, double& mean, double& var) const;

private:
  const SprAbsFilter* data_;
};

#endif
