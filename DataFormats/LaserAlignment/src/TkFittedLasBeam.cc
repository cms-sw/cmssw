#include "DataFormats/LaserAlignment/interface/TkFittedLasBeam.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <vector>

////////////////////////////////////////////////////////////////////////////////
TkFittedLasBeam::TkFittedLasBeam()
{
}

TkFittedLasBeam::TkFittedLasBeam(const TkLasBeam &lasBeam) : TkLasBeam(lasBeam)
{
}

////////////////////////////////////////////////////////////////////////////////
void TkFittedLasBeam::setParameters(unsigned int parametrisation,
				    const std::vector<Scalar> &params,
				    const AlgebraicSymMatrix &paramCovariance,
				    const std::vector<Scalar> &derivatives,
				    unsigned int firstFixedParam, float chi2)
{
  parametrisation_ = parametrisation;
  parameters_ = params;
  paramCovariance_ = paramCovariance;
  derivatives_ = derivatives;
  firstFixedParameter_ = firstFixedParam;
  chi2_ = chi2;

  // check integrity
  if (parameters_.size() != derivatives_.size() || firstFixedParameter_ > parameters_.size()
      || paramCovariance_.num_row() != firstFixedParameter_) {
    throw cms::Exception("BadInput")
      << "[TkFittedLasBeam::setParameters]  with inconsistent sizes: " 
      << parameters_.size() << " parameters, " << derivatives_.size()
      << " derivatives, " << " firstFixed = " << firstFixedParameter_ 
      << " cov. matrix size " << paramCovariance_.num_row() <<  ".";
  }
}

