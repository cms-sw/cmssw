#ifndef PhysicsTools_Utilities_NumericalIntegration_h
#define PhysicsTools_Utilities_NumericalIntegration_h
/* 
 * Numerical integration utilities
 *
 * \author: Luca Lista
 * Gauss Legendre and Gauss algorithms based on ROOT implementation
 *
 */
#include "Math/AllIntegrationTypes.h"
#include "Math/Functor.h"
#include "Math/Integrator.h"
#include <cmath>
#include <memory>
#include <vector>

namespace funct {

  template <typename F>
  double trapezoid_integral(const F& f, double min, double max, unsigned int samples) {
    const double l = max - min, delta = l / samples;
    double sum = 0;
    for (unsigned int i = 0; i < samples; ++i) {
      sum += f(min + (i + 0.5) * delta);
    }
    return sum * delta;
  }

  class TrapezoidIntegrator {
  public:
    TrapezoidIntegrator() : samples_(0) {}
    explicit TrapezoidIntegrator(unsigned int samples) : samples_(samples) {}
    template <typename F>
    double operator()(const F& f, double min, double max) const {
      return trapezoid_integral(f, min, max, samples_);
    }

  private:
    unsigned int samples_;
  };

  class GaussLegendreIntegrator {
  public:
    GaussLegendreIntegrator() : samples_(0) {}
    GaussLegendreIntegrator(unsigned int samples, double epsilon);
    template <typename F>
    double operator()(const F& f, double min, double max) const {
      a0 = 0.5 * (max + min);
      b0 = 0.5 * (max - min);
      result = 0.0;
      for (i = 0; i < samples_; ++i) {
        result += w[i] * f(a0 + b0 * x[i]);
      }

      return result * b0;
    }

  private:
    unsigned int samples_;
    std::vector<double> x, w;
    mutable double a0, b0, result;
    mutable unsigned int i;
  };

  class GaussIntegrator {
  public:
    GaussIntegrator() {}
    GaussIntegrator(double epsilon) : epsilon_(epsilon) {}
    template <typename F>
    double operator()(const F& f, double a, double b) const {
      h = 0;
      if (b == a)
        return h;
      aconst = kCST / std::abs(b - a);
      bb = a;
    CASE1:
      aa = bb;
      bb = b;
    CASE2:
      c1 = kHF * (bb + aa);
      c2 = kHF * (bb - aa);
      s8 = 0;
      for (i = 0; i < 4; ++i) {
        u = c2 * x[i];
        xx = c1 + u;
        f1 = f(xx);
        xx = c1 - u;
        f2 = f(xx);
        s8 += w[i] * (f1 + f2);
      }
      s16 = 0;
      for (i = 4; i < 12; ++i) {
        u = c2 * x[i];
        xx = c1 + u;
        f1 = f(xx);
        xx = c1 - u;
        f2 = f(xx);
        s16 += w[i] * (f1 + f2);
      }
      s16 = c2 * s16;
      if (std::abs(s16 - c2 * s8) <= epsilon_ * (1. + std::abs(s16))) {
        h += s16;
        if (bb != b)
          goto CASE1;
      } else {
        bb = c1;
        if (1. + aconst * std::abs(c2) != 1)
          goto CASE2;
        h = s8;
      }

      error_ = std::abs(s16 - c2 * s8);
      return h;
    }
    double error() const { return error_; }

  private:
    mutable double error_;
    double epsilon_;
    static const double x[12], w[12];
    static const double kHF;
    static const double kCST;
    mutable double h, aconst, bb, aa, c1, c2, u, s8, s16, f1, f2, xx;
    mutable unsigned int i;
  };

  struct RootIntegrator {
    RootIntegrator(ROOT::Math::IntegrationOneDim::Type type = ROOT::Math::IntegrationOneDim::kADAPTIVE,
                   double absTol = 1e-9,
                   double relTol = 1e-6,
                   unsigned int size = 1000,
                   unsigned int rule = 3)
        : type_(type),
          absTol_(absTol),
          relTol_(relTol),
          size_(size),
          rule_(rule),
          integrator_(new ROOT::Math::Integrator(type, absTol, relTol, size, rule)) {}
    RootIntegrator(const RootIntegrator& o) {
      type_ = o.type_;
      absTol_ = o.absTol_;
      relTol_ = o.relTol_;
      size_ = o.size_;
      rule_ = o.rule_;
      integrator_ = std::make_unique<ROOT::Math::Integrator>(type_, absTol_, relTol_, size_, rule_);
    }
    RootIntegrator& operator=(const RootIntegrator& o) {
      type_ = o.type_;
      absTol_ = o.absTol_;
      relTol_ = o.relTol_;
      size_ = o.size_;
      rule_ = o.rule_;
      integrator_ = std::make_unique<ROOT::Math::Integrator>(type_, absTol_, relTol_, size_, rule_);
      return *this;
    }
    template <typename F>
    double operator()(const F& f, double a, double b) const {
      ROOT::Math::Functor1D wrapper(&f, &F::operator());
      integrator_->SetFunction(wrapper);
      return integrator_->Integral(a, b);
    }

  private:
    ROOT::Math::IntegrationOneDim::Type type_;
    double absTol_, relTol_;
    unsigned int size_, rule_;
    mutable std::unique_ptr<ROOT::Math::Integrator> integrator_;
  };
}  // namespace funct

#endif
