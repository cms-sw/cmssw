#ifndef ElectroWeakAnalysis_ZMuMu_Gaussian_h
#define ElectroWeakAnalysis_ZMuMu_Gaussian_h

class Gaussian {
 public:
  Gaussian(double mean, double sigma);
  void setParameters(double mean, double sigma);
  double operator()(double x) const;
  double confidenceLevel99_7Interval() const;
  
 private:
  double mean_, sigma_;
  double mean2_, sigma2_;
};

#endif
