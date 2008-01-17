#ifndef PhysicsTools_Utilities_ZMuMu_BreitWigner_h
#define PhysicsTools_Utilities_ZMuMu_BreitWigner_h

class BreitWigner {
 public:
  BreitWigner(double m, double g);
  void setParameters(double m, double g);
  double operator()(double x) const;
  
 private:
  double m_, g_;
  double m2_, g2_, g2OverM2_;
};

#endif
