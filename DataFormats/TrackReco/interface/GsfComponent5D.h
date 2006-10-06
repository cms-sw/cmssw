#ifndef TrackReco_GsfComponent5D_h
#define TrackReco_GsfComponent5D_h

#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"

/// fixed size matrix
namespace reco {
class GsfComponent5D {
public:
  enum { dimension = 5 };
  typedef math::Vector<dimension>::type ParameterVector;
  typedef math::Error<dimension>::type CovarianceMatrix;
  GsfComponent5D () :
    weight_(0.) {}
  GsfComponent5D (const double& weight,
	      const ParameterVector& vector,
	      const CovarianceMatrix& matrix);
  double weight() const {return weight_;}
  const ParameterVector& parameters () const {return parameters_;}
  CovarianceMatrix& covariance (CovarianceMatrix& matrix) const;
private:
  double weight_;
  ParameterVector parameters_;
  double covariance_[dimension*(dimension+1)/2];
};
}
#endif
