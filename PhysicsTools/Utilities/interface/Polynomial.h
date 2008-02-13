#ifndef PhysicsTools_Utilities_Polynomial_h
#define PhysicsTools_Utilities_Polynomial_h
#include "boost/shared_ptr.hpp"

namespace function {
  template<unsigned int n>
  class Polynomial { 
  public:
    enum { arguments = 1 };
    enum { parameters = n + 1 }; 
    Polynomial(const double * c);
    Polynomial(const boost::shared_ptr<double> * c);
    void setParameters(const double * c);
    double operator()(double x) const;
  private:
    boost::shared_ptr<double> c0_;
    Polynomial<n-1> poly_;
  };
  
  template<unsigned int n>
  Polynomial<n>::Polynomial(const boost::shared_ptr<double> * c) : 
    c0_(c), poly_(c + 1) {
  }
  
  template<unsigned int n>
  Polynomial<n>::Polynomial(const double * c) : 
    c0_(new double(*c)), poly_(c + 1) {
  }
  
  template<unsigned int n>
  void Polynomial<n>::setParameters(const double * c) {
    *c0_ = *c; 
    poly_.setParameters(c + 1);
  }

  template<unsigned int n>
  double Polynomial<n>::operator()(double x) const {
    return *c0_ + x*poly_(x);
  }
  
  template<>
  class Polynomial<0> {
  public:
    enum { arguments = 1 };
    enum { parameters = 1 }; 
    Polynomial(const boost::shared_ptr<double> * c) : 
      c0_(*c) {
    }
    Polynomial(const double * c) : 
      c0_(new double(*c)) {
    }
    Polynomial(const boost::shared_ptr<double> c0) : 
      c0_(c0) {
    }
    Polynomial(double c0) : 
      c0_(new double(c0)) {
    }
    void setParameters(const double * c) {
      *c0_ = *c; 
    }
    void setParameters(double c0) {
      *c0_ = c0;
    }
    double operator()(double x) const {
      return *c0_;
    }
  private:
    boost::shared_ptr<double> c0_;
  };
  
  template<>
  class Polynomial<1> { 
  public:
    enum { arguments = 1 };
    enum { parameters = 2 }; 
    Polynomial(const boost::shared_ptr<double> * c) : 
      c0_(*c), poly_(c + 1) {
    }
    Polynomial(const double * c) : 
      c0_(new double(*c)), poly_(c + 1) {
    }
    void setParameters(const double * c) {
      *c0_ = *c; 
      poly_.setParameters(c + 1);
    }
    Polynomial(boost::shared_ptr<double> c0, boost::shared_ptr<double> c1) : 
      c0_(c0), poly_(c1) {
    }
    Polynomial(double c0, double c1) : 
      c0_(new double(c0)), poly_(c1) {
    }
    void setParameters(double c0, double c1) {
      *c0_ = c0; 
      poly_.setParameters(c1);
    }
    double operator()(double x) const {
      return *c0_ + x*poly_(x);
    }
  private:
    boost::shared_ptr<double> c0_;
    Polynomial<0> poly_;
  };
  
  template<>
  class Polynomial<2> { 
  public:
    enum { arguments = 1 };
    enum { parameters = 3 }; 
    Polynomial(const boost::shared_ptr<double> * c) : 
      c0_(*c), poly_(c + 1) {
    }
    Polynomial(const double * c) : 
      c0_(new double(*c)), poly_(c + 1) {
    }
    void setParameters(const double * c) {
      *c0_ = *c; 
      poly_.setParameters(c + 1);
    }
    Polynomial(boost::shared_ptr<double> c0, 
	       boost::shared_ptr<double> c1, 
	       boost::shared_ptr<double> c2) : c0_(c0), poly_(c1, c2) {
    }
    Polynomial(double c0, double c1, double c2) : 
      c0_(new double(c0) ), poly_(c1, c2) {
    }
    void setParameters(double c0, double c1, double c2) {
      *c0_ = c0; 
      poly_.setParameters(c1, c2);
    }
    double operator()(double x) const {
      return *c0_ + x*poly_(x);
    }
  private:
    boost::shared_ptr<double> c0_;
    Polynomial<1> poly_;
  };
}

#endif
