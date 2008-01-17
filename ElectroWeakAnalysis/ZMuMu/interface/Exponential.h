#ifndef ElectroWeakAnalysis_ZMuMu_Exponential_h
#define ElectroWeakAnalysis_ZMuMu_Exponential_h

class Exponential {
 public:
  Exponential(double parameter);
  void setParameters(double parameter);
  double operator()(double x) const;
  
 private:
  double parameter_;
};

#endif
