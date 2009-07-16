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
				    const AlgebraicMatrix &derivatives,
				    unsigned int firstFixedParam, float chi2)
{
  parametrisation_ = parametrisation;
  parameters_ = params;
  paramCovariance_ = paramCovariance;
  derivatives_ = derivatives;
  firstFixedParameter_ = firstFixedParam;
  chi2_ = chi2;

  // check integrity
//   if (parameters_.size() != static_cast<unsigned int>(derivatives_.num_col()) || firstFixedParameter_ > parameters_.size()
//       || static_cast<unsigned int>(paramCovariance_.num_row()) != firstFixedParameter_) {
//     throw cms::Exception("BadInput")
//       << "[TkFittedLasBeam::setParameters]  with inconsistent sizes: " 
//       << parameters_.size() << " parameters, " << derivatives_.num_col()
//       << " derivatives, " << " firstFixed = " << firstFixedParameter_ 
//       << " cov. matrix size " << paramCovariance_.num_row() <<  ".";
//   }
}

