#ifndef PhysicsTools_Utilities_Polynomial_h
#define PhysicsTools_Utilities_Polynomial_h
#include <utility>

#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "boost/shared_ptr.hpp"

namespace funct {
  template<unsigned int n>
  class Polynomial { 
  public:
    Polynomial(const double * c);
    Polynomial(const boost::shared_ptr<double> * c);
    Polynomial(const Parameter * p);
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
  Polynomial<n>::Polynomial(const Parameter * c) : 
    c0_(c->ptr()), poly_(c + 1) {
  }
  
  
  template<unsigned int n>
  Polynomial<n>::Polynomial(const double * c) : 
    c0_(new double(*c)), poly_(c + 1) {
  }

  template<unsigned int n>
  double Polynomial<n>::operator()(double x) const {
    return *c0_ + x*poly_(x);
  }
  
  template<>
  class Polynomial<0> {
  public:
    Polynomial(const boost::shared_ptr<double> * c) : 
      c0_(*c) {
    }
    Polynomial(const Parameter * c) : 
      c0_(c->ptr()) {
    }
    Polynomial(const double * c) : 
      c0_(new double(*c)) {
    }
    Polynomial(boost::shared_ptr<double> c0) : 
      c0_(std::move(c0)) {
    }
    Polynomial(const Parameter & c0) : 
      c0_(c0.ptr()) {
    }
    Polynomial(double c0) : 
      c0_(new double(c0)) {
    }
    double operator()(double x) const {
      return *c0_;
    }
    double operator()() const {
      return *c0_;
    }
  private:
    boost::shared_ptr<double> c0_;
  };
  
  template<>
  class Polynomial<1> { 
  public:
    Polynomial(const boost::shared_ptr<double> * c) : 
      c0_(*c), poly_(c + 1) {
    }
    Polynomial(const Parameter * c) : 
      c0_(c->ptr()), poly_(c + 1) {
    }
    Polynomial(const double * c) : 
      c0_(new double(*c)), poly_(c + 1) {
    }
    Polynomial(boost::shared_ptr<double> c0, boost::shared_ptr<double> c1) : 
      c0_(std::move(c0)), poly_(std::move(c1)) {
    }
    Polynomial(const Parameter& c0, const Parameter& c1) : 
      c0_(c0.ptr()), poly_(c1.ptr()) {
    }
    Polynomial(double c0, double c1) : 
      c0_(new double(c0)), poly_(c1) {
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
    Polynomial(const boost::shared_ptr<double> * c) : 
      c0_(*c), poly_(c + 1) {
    }
    Polynomial(const Parameter * c) :
      c0_(c->ptr()), poly_(c + 1) {
    }
    Polynomial(const double * c) : 
      c0_(new double(*c)), poly_(c + 1) {
    }
    Polynomial(boost::shared_ptr<double> c0, 
	       boost::shared_ptr<double> c1, 
	       boost::shared_ptr<double> c2) : c0_(std::move(c0)), poly_(std::move(c1), std::move(c2)) {
    }
    Polynomial(const Parameter &c0, 
	       const Parameter &c1, 
	       const Parameter &c2) : c0_(c0.ptr()), poly_(c1, c2) {
    }
    Polynomial(double c0, double c1, double c2) : 
      c0_(new double(c0) ), poly_(c1, c2) {
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
