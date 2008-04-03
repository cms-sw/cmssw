#ifndef PhysicsTools_Utilities_Power_h
#define PhysicsTools_Utilities_Power_h
#include <boost/static_assert.hpp>
#include <cmath>

namespace funct {
  template<typename A, typename B, unsigned int args = A::arguments>
  struct PowerStruct { 
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = args;
  };

  template<typename A, typename B>
  struct PowerStruct<A, B, 0> { 
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 0;
    PowerStruct(const A & a, const B & b) : _1(a), _2(b) { }
    double operator()() const {
      return pow(_1(), _2());
    }
    operator double() const {
      return pow(_1(), _2());
    }
    A _1; 
    B _2;
  };

  template<typename A, typename B>
  struct PowerStruct<A, B, 1> { 
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 1;
    PowerStruct(const A & a, const B & b) : _1(a), _2(b) { }
    double operator()(double x) const {
      return pow(_1(x), _2(x));
    }
    A _1; 
    B _2;
  };

  template<typename A, typename B>
  struct PowerStruct<A, B, 2> { 
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 2;
    PowerStruct(const A & a, const B & b) : _1(a), _2(b) { }
    double operator()(double x, double y) const {
      return pow(_1(x, y), _2(x, y));
    }
    A _1; 
    B _2;
  };

  template<typename A, typename B>
  struct Power {
    typedef PowerStruct<A, B> type;
    static type combine(const A& a, const B& b) { 
      return type(a, b);
    }
  };

  template<typename A, typename B>
  inline typename Power<A, B>::type operator^(const A& a, const B& b) {
    return Power<A, B>::combine(a, b);
  }

}

#endif
