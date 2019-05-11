#ifndef CALIBCALORIMETRY_HCALALGOS_HCALSIPMNONLINEARITY_H
#define CALIBCALORIMETRY_HCALALGOS_HCALSIPMNONLINEARITY_H 1

#include <vector>
#include <cassert>

class HcalSiPMnonlinearity {
public:
  HcalSiPMnonlinearity(const std::vector<float>& pars) {
    assert(pars.size() == 3);
    c0 = (double)pars[0];
    b1 = (double)pars[1];
    a2 = (double)pars[2];
  }

  // for Reco
  inline double getRecoCorrectionFactor(double inpixelsfired) const {
    double x = inpixelsfired;
    return (a2 * x * x + b1 * x + c0);
  }

  // for Sim
  int getPixelsFired(int inpes) const;

private:
  // quadratic coefficients
  double a2, b1, c0;
};

#endif
