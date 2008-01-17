#ifndef PhysicsTools_Utilities_Exponential_h
#define PhysicsTools_Utilities_Exponential_h

class Exponential {
 public:
  Exponential(double lambda);
  void setParameters(double lambda);
  double operator()(double x) const;
  
 private:
  double lambda_;
};

#endif
