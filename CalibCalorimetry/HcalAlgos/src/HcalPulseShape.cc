#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShape.h"

HcalPulseShape::HcalPulseShape() {
  nbin_=0;
}

HcalPulseShape::HcalPulseShape(const std::vector<double>& shape, unsigned nbin) :
  shape_(shape.begin(),shape.begin()+nbin),
  nbin_(nbin)
{
}


void HcalPulseShape::setNBin(int n) {
  nbin_=n;
  shape_=std::vector<float>(n,0.0f);
}

void HcalPulseShape::setShapeBin(int i, float f) {
  if (i>=0 && i<nbin_) shape_[i]=f;
}

float HcalPulseShape::operator()(double t) const {
  // shape is in 1 ns steps
  return at(t);
}

float HcalPulseShape::at(double t) const {
  // shape is in 1 ns steps
  int i=(int)(t+0.5);
  float rv=0;
  if (i>=0 && i<nbin_) rv=shape_[i];
  return rv;
}

float HcalPulseShape::integrate(double t1, double t2) const {
  static const float int_delta_ns = 0.05f;
  double intval = 0.0;

  for (double t = t1; t < t2; t+= int_delta_ns) {
    float loedge = at(t);
    float hiedge = at(t+int_delta_ns);
    intval += (loedge+hiedge)*int_delta_ns/2.0;
  }
  return (float)intval;
}

