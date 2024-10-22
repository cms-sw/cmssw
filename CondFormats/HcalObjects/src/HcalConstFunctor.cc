#include "CondFormats/HcalObjects/interface/HcalConstFunctor.h"

HcalConstFunctor::HcalConstFunctor() : value_(0.0) {}

HcalConstFunctor::HcalConstFunctor(const double d) : value_(d) {}

double HcalConstFunctor::operator()(double) const { return value_; }

BOOST_CLASS_EXPORT_IMPLEMENT(HcalConstFunctor)
