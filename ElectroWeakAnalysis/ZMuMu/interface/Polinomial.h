#ifndef ElectroWeakAnalysis_ZMuMu_Polinomial_h
#define ElectroWeakAnalysis_ZMuMu_Polinomial_h

class Polinomial {
 public:
  Polinomial(double a_0, double a_1, double a_2);
  void setParameters(double a_0, double a_1, double a_2);
  double operator()(double x) const;
  
 private:
  double a_0_, a_1_, a_2_;
};

#endif
