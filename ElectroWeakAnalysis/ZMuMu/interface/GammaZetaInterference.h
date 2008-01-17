#ifndef ElectroWeakAnalysis_ZMuMu_GammaZetaInterference_h
#define ElectroWeakAnalysis_ZMuMu_GammaZetaInterference_h

class GammaZetaInterference {
 public:
  GammaZetaInterference(double m, double g);
  void setParameters(double m, double g);
  double operator()(double x) const;
  
 private:
  double m_, g_;
  double m2_, g2_, g2OverM2_;
};

#endif
