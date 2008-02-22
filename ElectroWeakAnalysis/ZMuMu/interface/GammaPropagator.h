#ifndef ElectroWeakAnalysis_ZMuMu_GammaPropagator_h
#define ElectroWeakAnalysis_ZMuMu_GammaPropagator_h

class GammaPropagator {
 public:
  GammaPropagator();
  void setParameters();
  double operator()(double x) const;
  
 private:
  double g_;
};

#endif
