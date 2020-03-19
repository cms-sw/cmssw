#include "CalibCalorimetry/HcalAlgos/interface/HcalShapeIntegrator.h"

#include <iostream>
HcalShapeIntegrator::HcalShapeIntegrator(const HcalPulseShapes::Shape* shape) : nbin_(shape->nbins()), v_(nbin_, 0.) {
  for (int t = 0; t < nbin_; ++t) {
    double amount = shape->at(t);
    for (int ibin = t; ibin < nbin_; ++ibin) {
      // v_ holds the cumulative integral
      v_[ibin] += amount;
    }
  }
}

float HcalShapeIntegrator::at(double t) const {
  // shape is in 1 ns steps
  // I round down to match the old algorithm
  int i = (int)(t - 0.5);
  float rv = 0;
  if (i < 0) {
    rv = 0.;
  } else if (i >= nbin_) {
    rv = v_.back();
  } else {
    rv = v_[i];
    // maybe interpolate
    // assume 1 ns bins
    float f = (t - 0.5 - i);
    if (++i < nbin_ && f > 0) {
      rv = rv * (1. - f) + v_[i] * f;
    }
  }
  return rv;
}

float HcalShapeIntegrator::operator()(double startTime, double stopTime) const { return at(stopTime) - at(startTime); }
