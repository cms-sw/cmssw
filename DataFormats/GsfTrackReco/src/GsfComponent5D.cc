#include "DataFormats/GsfTrackReco/interface/GsfComponent5D.h"

using namespace reco;

GsfComponent5D::GsfComponent5D(const double& weight, const ParameterVector& parameters, const CovarianceMatrix& matrix)
    : weight_(weight), parameters_(parameters) {
  float* data(covariance_);
  typedef unsigned int index;
  for (index i = 0; i < dimension; ++i)
    for (index j = 0; j <= i; ++j)
      *(data++) = matrix(i, j);
}

GsfComponent5D::CovarianceMatrix& GsfComponent5D::covariance(CovarianceMatrix& matrix) const {
  const float* data(covariance_);
  typedef unsigned int index;
  for (index i = 0; i < dimension; ++i)
    for (index j = 0; j <= i; ++j)
      matrix(i, j) = *(data++);
  return matrix;
}
