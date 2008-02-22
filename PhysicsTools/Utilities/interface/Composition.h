#ifndef PhysicsTools_Utilities_Composition_h
#define PhysicsTools_Utilities_Composition_h
#include <boost/static_assert.hpp>

namespace function {
  template<typename A, typename B, unsigned int args = B::arguments>
  class Composition { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == 1);
    static const unsigned int arguments = args;
  };

  template<typename A, typename B>
  class Composition<A, B, 0> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 0;
    Composition(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()() const {
      return a_(b_());
    }
  private:
    A a_; 
    B b_;
  };

  template<typename A, typename B>
  class Composition<A, B, 1> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 1;
    Composition(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()(double x) const {
      return a_(b_(x));
    }
  private:
    A a_; 
    B b_;
  };

  template<typename A, typename B>
  class Composition<A, B, 2> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 2;
    Composition(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()(double x, double y) const {
      return a_(b_(x, y));
    }
  private:
    A a_; 
    B b_;
  };

}

template<typename A, typename B>
function::Composition<A, B> operator%(const A& a, const B& b) {
  return function::Composition<A, B>(a, b);
}

#endif
