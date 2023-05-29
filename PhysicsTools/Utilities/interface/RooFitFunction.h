#ifndef PhysicsTools_Utilities_RooFitFunction_h
#define PhysicsTools_Utilities_RooFitFunction_h

#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "RooAbsReal.h"
#include "RooListProxy.h"
#include "RooRealProxy.h"

#include <iostream>
#include <vector>

namespace root {
  struct no_args;

  template <typename X, typename Expr>
  class RooFitFunction : public RooAbsReal {
  public:
    RooFitFunction(const RooFitFunction<X, Expr>& other, const char* name = nullptr)
        : RooAbsReal(other, name),
          e_(other.e_),
          x_(X::name(), this, other.x_),
          parsPtrs_{other.parsPtrs_},
          parsArgs_{"!pars", this, other.parsArgs_} {}
    RooFitFunction(const char* name, const char* title, const Expr& e, RooAbsReal& x)
        : RooAbsReal(name, title),
          e_(e),
          x_(X::name(), X::name(), this, x),
          parsArgs_("!pars", "List of parameters", this) {}
    RooFitFunction(
        const char* name, const char* title, const Expr& e, RooAbsReal& x, RooAbsReal& rA, funct::Parameter& a)
        : RooFitFunction{name, title, e, x} {
      add(rA, a);
    }
    RooFitFunction(const char* name,
                   const char* title,
                   const Expr& e,
                   RooAbsReal& x,
                   RooAbsReal& rA,
                   funct::Parameter& a,
                   RooAbsReal& rB,
                   funct::Parameter& b)
        : RooFitFunction{name, title, e, x, rA, a} {
      add(rB, b);
    }
    RooFitFunction(const char* name,
                   const char* title,
                   const Expr& e,
                   RooAbsReal& x,
                   RooAbsReal& rA,
                   funct::Parameter& a,
                   RooAbsReal& rB,
                   funct::Parameter& b,
                   RooAbsReal& rC,
                   funct::Parameter& c)
        : RooFitFunction{name, title, e, x, rA, a, rB, b} {
      add(rC, c);
    }
    void add(RooAbsReal& rA, funct::Parameter& a) {
      parsPtrs_.emplace_back(a.ptr());
      parsArgs_.add(rA);
    }
    TObject* clone(const char* newName) const override { return new RooFitFunction<X, Expr>(*this, newName); }

  private:
    Expr e_;
    RooRealProxy x_;
    std::vector<std::shared_ptr<double>> parsPtrs_;
    RooListProxy parsArgs_;
    Double_t evaluate() const override {
      X::set(x_);
      for (std::size_t i = 0; i < parsPtrs_.size(); ++i) {
        *(parsPtrs_[i]) = static_cast<RooAbsReal const&>(parsArgs_[i]).getVal(parsArgs_.nset());
      }
      return e_();
    }
  };

}  // namespace root

#endif
