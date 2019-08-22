#include <algorithm>
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/HcalPolynomialFunctor.h"

HcalPolynomialFunctor::HcalPolynomialFunctor() : shift_(0.0), xmin_(-DBL_MAX), xmax_(DBL_MAX), outOfRangeValue_(0.0) {}

HcalPolynomialFunctor::HcalPolynomialFunctor(const std::vector<double>& coeffs,
                                             const double shift,
                                             const double xmin,
                                             const double xmax,
                                             const double outOfRangeValue)
    : coeffs_(coeffs), shift_(shift), xmin_(xmin), xmax_(xmax), outOfRangeValue_(outOfRangeValue) {}

double HcalPolynomialFunctor::operator()(const double x) const {
  double result = outOfRangeValue_;
  if (x >= xmin_ && x <= xmax_) {
    result = 0.0;
    if (!coeffs_.empty()) {
      const double* a = &coeffs_[0];
      const double y = x + shift_;
      for (int deg = coeffs_.size() - 1; deg >= 0; --deg) {
        result *= y;
        result += a[deg];
      }
    }
  }
  return result;
}

BOOST_CLASS_EXPORT_IMPLEMENT(HcalPolynomialFunctor)
