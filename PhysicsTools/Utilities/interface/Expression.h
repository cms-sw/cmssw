#ifndef PhysicsTools_Utilities_Expression_h
#define PhysicsTools_Utilities_Expression_h
#include "PhysicsTools/Utilities/interface/FunctionsIO.h"
#include <ostream>
#include <memory>

namespace funct {

  struct AbsExpression {
    virtual ~AbsExpression() { }
    virtual double operator()() const = 0;
    virtual AbsExpression * clone() const = 0;
    virtual std::ostream& print(std::ostream& cout) const = 0;
  };

  template<typename F>
  struct ExpressionT : public AbsExpression {
    inline ExpressionT(const F& f) : _f(f) {}
    virtual ~ExpressionT() { }
    virtual double operator()() const { return _f(); }
    virtual AbsExpression * clone() const { return new ExpressionT<F>(_f); }
    virtual std::ostream& print(std::ostream& cout) const { return cout << _f; }
  private:
    F _f;
  };

 struct Expression {
   inline Expression() { }
   template<typename F>
   inline Expression(const F& f) : _f(new ExpressionT<F>(f)) { }
   inline Expression(const Expression& e) : _f(e._f->clone()) { }
   inline Expression& operator=(const Expression& e) { _f.reset(e._f->clone()); return *this; }
   inline double operator()() const { return (*_f)(); }
   inline std::ostream& print(std::ostream& cout) const { return _f->print(cout); }
 private:
   std::unique_ptr<AbsExpression> _f;
 };

 inline std::ostream& operator<<(std::ostream& cout, const Expression& e) { 
   e.print(cout); return cout; 
 }

  struct AbsFunctExpression {
    virtual ~AbsFunctExpression() { }
    virtual double operator()(double x) const = 0;
    virtual AbsFunctExpression * clone() const = 0;
  };

  template<typename F>
  struct FunctExpressionT : public AbsFunctExpression {
    inline FunctExpressionT(const F& f) : _f(f) {}
    virtual ~FunctExpressionT() { }
    virtual double operator()(double x) const { return _f(x); }
    virtual AbsFunctExpression * clone() const { return new FunctExpressionT<F>(_f); }
  private:
    F _f;
  };

 struct FunctExpression {
   inline FunctExpression() { }
   template<typename F>
   inline FunctExpression(const F& f) : _f(new FunctExpressionT<F>(f)) { }
   inline FunctExpression(const FunctExpression& e) : _f(e._f->clone()) { }
   inline FunctExpression& operator=(const FunctExpression& e) { _f.reset(e._f->clone()); return *this; }
   inline double operator()(double x) const { return (*_f)(x); }
 private:
   std::unique_ptr<AbsFunctExpression> _f;
 };

}

#endif
