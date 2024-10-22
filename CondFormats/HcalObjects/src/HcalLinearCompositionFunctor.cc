#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/HcalLinearCompositionFunctor.h"

HcalLinearCompositionFunctor::HcalLinearCompositionFunctor(std::shared_ptr<AbsHcalFunctor> p,
                                                           const double ia,
                                                           const double ib)
    : other_(p), a_(ia), b_(ib) {
  if (!other_.get())
    throw cms::Exception(
        "In HcalLinearCompositionFunctor constructor: "
        "can not use a null pointer to another functor");
}

double HcalLinearCompositionFunctor::operator()(const double x) const { return a_ * (*other_)(x) + b_; }

BOOST_CLASS_EXPORT_IMPLEMENT(HcalLinearCompositionFunctor)
