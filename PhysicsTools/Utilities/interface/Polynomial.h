#ifndef PhysicsTools_Utilities_Polynomial_h
#define PhysicsTools_Utilities_Polynomial_h
#include "PhysicsTools/Utilities/interface/Parameter.h"

namespace funct {
  template <unsigned int n>
  class Polynomial {
  public:
    Polynomial(const double *c);
    Polynomial(const std::shared_ptr<double> *c);
    Polynomial(const Parameter *p);
    double operator()(double x) const;

  private:
    std::shared_ptr<double> c0_;
    Polynomial<n - 1> poly_;
  };

  template <unsigned int n>
  Polynomial<n>::Polynomial(const std::shared_ptr<double> *c) : c0_(*c), poly_(c + 1) {}
  template <unsigned int n>
  Polynomial<n>::Polynomial(const Parameter *c) : c0_(c->ptr()), poly_(c + 1) {}

  template <unsigned int n>
  Polynomial<n>::Polynomial(const double *c) : c0_(new double(*c)), poly_(c + 1) {}

  template <unsigned int n>
  double Polynomial<n>::operator()(double x) const {
    return *c0_ + x * poly_(x);
  }

  template <>
  class Polynomial<0> {
  public:
    Polynomial(const std::shared_ptr<double> *c) : c0_(*c) {}
    Polynomial(const Parameter *c) : c0_(c->ptr()) {}
    Polynomial(const double *c) : c0_(new double(*c)) {}
    Polynomial(std::shared_ptr<double> c0) : c0_(c0) {}
    Polynomial(const Parameter &c0) : c0_(c0.ptr()) {}
    Polynomial(double c0) : c0_(new double(c0)) {}
    double operator()(double x) const { return *c0_; }
    double operator()() const { return *c0_; }

  private:
    std::shared_ptr<double> c0_;
  };

  template <>
  class Polynomial<1> {
  public:
    Polynomial(const std::shared_ptr<double> *c) : c0_(*c), poly_(c + 1) {}
    Polynomial(const Parameter *c) : c0_(c->ptr()), poly_(c + 1) {}
    Polynomial(const double *c) : c0_(new double(*c)), poly_(c + 1) {}
    Polynomial(std::shared_ptr<double> c0, std::shared_ptr<double> c1) : c0_(c0), poly_(c1) {}
    Polynomial(const Parameter &c0, const Parameter &c1) : c0_(c0.ptr()), poly_(c1.ptr()) {}
    Polynomial(double c0, double c1) : c0_(new double(c0)), poly_(c1) {}
    double operator()(double x) const { return *c0_ + x * poly_(x); }

  private:
    std::shared_ptr<double> c0_;
    Polynomial<0> poly_;
  };

  template <>
  class Polynomial<2> {
  public:
    Polynomial(const std::shared_ptr<double> *c) : c0_(*c), poly_(c + 1) {}
    Polynomial(const Parameter *c) : c0_(c->ptr()), poly_(c + 1) {}
    Polynomial(const double *c) : c0_(new double(*c)), poly_(c + 1) {}
    Polynomial(std::shared_ptr<double> c0, std::shared_ptr<double> c1, std::shared_ptr<double> c2)
        : c0_(c0), poly_(c1, c2) {}
    Polynomial(const Parameter &c0, const Parameter &c1, const Parameter &c2) : c0_(c0.ptr()), poly_(c1, c2) {}
    Polynomial(double c0, double c1, double c2) : c0_(new double(c0)), poly_(c1, c2) {}
    double operator()(double x) const { return *c0_ + x * poly_(x); }

  private:
    std::shared_ptr<double> c0_;
    Polynomial<1> poly_;
  };
}  // namespace funct

#endif
