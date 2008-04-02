#ifndef PhysicsTools_Utilities_Difference_h
#define PhysicsTools_Utilities_Difference_h
#include <boost/static_assert.hpp>

namespace function {
  template<typename A, typename B, unsigned int args = A::arguments>
  class DifferenceStruct { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = args;
  };

  template<typename A, typename B>
  class DifferenceStruct<A, B, 0> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 0;
    DifferenceStruct(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()() const {
      return a_() - b_();
    }
    operator double() const {
      return a_() - b_();
    }
  private:
    A a_; 
    B b_;
  };

  template<typename A, typename B>
  class DifferenceStruct<A, B, 1> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 1;
    DifferenceStruct(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()(double x) const {
      return a_(x) - b_(x);
    }
  private:
    A a_; 
    B b_;
  };

  template<typename A, typename B>
  class DifferenceStruct<A, B, 2> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 2;
    DifferenceStruct(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()(double x, double y) const {
      return a_(x, y) - b_(x, y);
    }
  private:
    A a_; 
    B b_;
  };

  template<typename A, typename B>
  struct Difference {
    typedef DifferenceStruct<A, B> type;
    static type combine(const A& a, const B& b) { return type(a, b); }
  };
}

template<typename A, typename B>
inline typename function::Difference<A, B>::type operator-(const A& a, const B& b) {
  return function::Difference<A, B>::combine(a, b);
}

#include "PhysicsTools/Utilities/interface/Number.h"

#include "PhysicsTools/Utilities/interface/Constant.h"


template<typename A>
typename function::Difference<A, function::Constant>::type 
inline operator-(const A& a, const function::Parameter& b) {
  return function::Difference<A, function::Constant>::combine(a, function::Constant(b));
}

template<typename B>
typename function::Difference<function::Number, B>::type 
inline operator-(double a, const B& b) {
  return function::Difference<function::Number, B>::combine(function::Number(a), b);
}

function::Difference<function::Number, function::Constant>::type
inline operator-(double a, const function::Parameter& b) {
  return function::Difference<function::Number, 
    function::Constant>::combine(function::Number(a), function::Constant(b));
}

#endif
