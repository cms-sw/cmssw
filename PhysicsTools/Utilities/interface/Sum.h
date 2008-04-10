#ifndef PhysicsTools_Utilities_Sum_h
#define PhysicsTools_Utilities_Sum_h
#include <boost/static_assert.hpp>

namespace function {
  template<typename A, typename B, unsigned int args = A::arguments>
  class Sum { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
  static const unsigned int arguments = args;
  };

  template<typename A, typename B>
  class Sum<A, B, 0> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 0;
    Sum(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()() const {
      return a_() + b_();
    }
  private:
    A a_; 
    B b_;
  };

  template<typename A, typename B>
  class Sum<A, B, 1> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 1;
    Sum(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()(double x) const {
      return a_(x) + b_(x);
    }
  private:
    A a_; 
    B b_;
  };

  template<typename A, typename B>
  class Sum<A, B, 2> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 2;
    Sum(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()(double x, double y) const {
      return a_(x, y) + b_(x, y);
    }
  private:
    A a_; 
    B b_;
  };
}

template<typename A, typename B>
function::Sum<A, B> operator+(const A& a, const B& b) {
  return function::Sum<A, B>(a, b);
}

#endif
