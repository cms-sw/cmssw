#include "DataFormats/Alignment/interface/TkFittedLasBeam.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <vector>

////////////////////////////////////////////////////////////////////////////////
TkFittedLasBeam::TkFittedLasBeam() {}

TkFittedLasBeam::TkFittedLasBeam(const TkLasBeam &lasBeam) : TkLasBeam(lasBeam) {}

////////////////////////////////////////////////////////////////////////////////
void TkFittedLasBeam::setParameters(unsigned int parametrisation,
                                    const std::vector<Scalar> &params,
                                    const AlgebraicSymMatrix &paramCovariance,
                                    const AlgebraicMatrix &derivatives,
                                    unsigned int firstFixedParam,
                                    float chi2) {
  parametrisation_ = parametrisation;
  parameters_ = params;
  paramCovariance_ = paramCovariance;
  derivatives_ = derivatives;
  firstFixedParameter_ = firstFixedParam;
  chi2_ = chi2;

  // check integrity
  if (parameters_.size() != static_cast<unsigned int>(derivatives_.num_col())         // # parameters
      || static_cast<unsigned int>(derivatives_.num_row()) != this->getData().size()  // # hits
      || firstFixedParameter_ > parameters_.size()                                    // index 'fixed' might be the
      || static_cast<unsigned int>(paramCovariance_.num_row()) != firstFixedParameter_) {
    throw cms::Exception("BadInput")
        << "[TkFittedLasBeam::setParameters] with inconsistent sizes: (parametrisation " << parametrisation << "):\n"
        << parameters_.size() << " parameters,\n"
        << derivatives_.num_row() << "x" << derivatives_.num_col() << " derivatives,\n"
        << "firstFixed = " << firstFixedParameter_ << " (i.e. "
        << static_cast<int>(parameters_.size()) - static_cast<int>(firstFixedParameter_)
        << " global parameters),\n"  // cast on line before to allow difference to be < 0, [un]signed!
        << "cov. matrix size " << paramCovariance_.num_row() << ".\n";
  }
}
