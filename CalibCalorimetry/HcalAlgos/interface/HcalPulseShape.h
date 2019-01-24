#ifndef HcalAlgos_HcalPulseShape_h
#define HcalAlgos_HcalPulseShape_h

#include<vector>

class HcalPulseShape {
public:
  HcalPulseShape();
  HcalPulseShape(const std::vector<double>&,unsigned);
  void setNBin(int n);
  void setShapeBin(int i, float f);
  float operator()(double time) const;
  float at(double time) const;
  float integrate(double tmin, double tmax) const;
  int nbins() const {return nbin_;}
private:
  std::vector<float> shape_;
  int nbin_;
};

#endif
