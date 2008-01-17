#ifndef PhysicsTools_Utilities_Gaussian_h
#define PhysicsTools_Utilities_Gaussian_h

class Gaussian {
 public:
  Gaussian(double mean, double sigma);
  void setParameters(double mean, double sigma);
  double operator()(double x) const;
  
 private:
  double mean_, sigma_;
};

#endif
