#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/HcalChebyshevFunctor.h"

HcalChebyshevFunctor::HcalChebyshevFunctor() : xmin_(-1.0), xmax_(1.0), outOfRangeValue_(0.0) {}

HcalChebyshevFunctor::HcalChebyshevFunctor(const std::vector<double>& coeffs,
                                           const double xmin,
                                           const double xmax,
                                           const double outOfRangeValue)
    : coeffs_(coeffs), xmin_(xmin), xmax_(xmax), outOfRangeValue_(outOfRangeValue) {
  if (xmin_ >= xmax_)
    throw cms::Exception("In HcalChebyshevFunctor constructor: invalid interval specification");
}

double HcalChebyshevFunctor::operator()(const double y) const {
  if (!(y >= xmin_ && y <= xmax_))
    return outOfRangeValue_;

  if (coeffs_.empty())
    return 0.0;

  const double x = 2.0 * (y - xmin_) / (xmax_ - xmin_) - 1.0;
  const double* a = &coeffs_[0];
  const double twox = 2.0 * x;

  // Clenshaw recursion
  double rp2 = 0.0, rp1 = 0.0, r = 0.0;
  for (unsigned k = coeffs_.size() - 1; k > 0U; --k) {
    r = twox * rp1 - rp2 + a[k];
    rp2 = rp1;
    rp1 = r;
  }
  return x * rp1 - rp2 + a[0];
}

BOOST_CLASS_EXPORT_IMPLEMENT(HcalChebyshevFunctor)
