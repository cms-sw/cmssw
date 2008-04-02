#ifndef PhysicsTools_Utilities_Product_h
#define PhysicsTools_Utilities_Product_h
#include <boost/static_assert.hpp>

namespace funct {
  template<typename A, typename B, unsigned int args = A::arguments>
  struct ProductStruct { 
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = args;
  };

  template<typename A, typename B>
  struct ProductStruct<A, B, 0> { 
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 0;
    ProductStruct(const A & a, const B & b) : _1(a), _2(b) { }
    double operator()() const {
      return _1() * _2();
    }
    operator double() const {
      return _1() * _2();
    }
    A _1; 
    B _2;
  };

  template<typename A, typename B>
  struct ProductStruct<A, B, 1> { 
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 1;
    ProductStruct(const A & a, const B & b) : _1(a), _2(b) { }
    double operator()(double x) const {
      return _1(x) * _2(x);
    }
    A _1; 
    B _2;
  };

  template<typename A, typename B>
  struct ProductStruct<A, B, 2> { 
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 2;
    ProductStruct(const A & a, const B & b) : _1(a), _2(b) { }
    double operator()(double x, double y) const {
      return _1(x, y) * _2(x, y);
    }
    A _1; 
    B _2;
  };

  template<typename A, typename B>
  struct Product {
    typedef ProductStruct<A, B> type;
    static type combine(const A& a, const B& b) { 
      return type(a, b);
    }
  };
}

#endif
