#ifndef PhysicsTools_Utilities_Polynomial_h
#define PhysicsTools_Utilities_Polynomial_h

template<unsigned int n>
class Polynomial { 
public:
  Polynomial(const double * c) : c0_(*c), poly_(c + 1) {
  }
  void setParameters(const double * c) {
    c0_ = *c; 
    poly_.setParameters(c + 1);
  }
  double operator()(double x) {
    return c0_ + x*poly_(x);
  }
private:
  double c0_;
  Polynomial<n-1> poly_;
};

template<>
class Polynomial<0> {
public:
  Polynomial(const double * c) {
    setParameters(c);
  }
  void setParameters(const double * c) {
    c0_ = *c;
  }
  double operator()(double x) {
    return c0_;
  }
private:
  double c0_;
};



#endif
