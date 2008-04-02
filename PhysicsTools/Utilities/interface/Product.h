#ifndef PhysicsTools_Utilities_Product_h
#define PhysicsTools_Utilities_Product_h
#include <boost/static_assert.hpp>

namespace funct {
  template<typename A, typename B, unsigned int args = A::arguments>
  class ProductStruct { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = args;
  };

  template<typename A, typename B>
  class ProductStruct<A, B, 0> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 0;
    ProductStruct(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()() const {
      return a_() * b_();
    }
    operator double() const {
      return a_() * b_();
    }
  private:
    A a_; 
    B b_;
  };

  template<typename A, typename B>
  class ProductStruct<A, B, 1> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 1;
    ProductStruct(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()(double x) const {
      return a_(x) * b_(x);
    }
  private:
    A a_; 
    B b_;
  };

  template<typename A, typename B>
  class ProductStruct<A, B, 2> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 2;
    ProductStruct(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()(double x, double y) const {
      return a_(x, y) * b_(x, y);
    }
  private:
    A a_; 
    B b_;
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
