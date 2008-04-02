#ifndef PhysicsTools_Utilities_Composition_h
#define PhysicsTools_Utilities_Composition_h
#include <boost/static_assert.hpp>

namespace funct {
  template<typename A, typename B, unsigned int args = B::arguments>
  class CompositionStruct { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == 1);
    static const unsigned int arguments = args;
  };

  template<typename A, typename B>
  class CompositionStruct<A, B, 0> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 0;
    CompositionStruct(const A & a, const B & b) : _1(a), _2(b) { }
    double operator()() const {
      return _1(_2());
    }
  private:
    A _1; 
    B _2;
  };

  template<typename A, typename B>
  class CompositionStruct<A, B, 1> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 1;
    CompositionStruct(const A & a, const B & b) : _1(a), _2(b) { }
    double operator()(double x) const {
      return _1(_2(x));
    }
  private:
    A _1; 
    B _2;
  };

  template<typename A, typename B>
  class CompositionStruct<A, B, 2> { 
  public:
    BOOST_STATIC_ASSERT(A::arguments == B::arguments);
    static const unsigned int arguments = 2;
    CompositionStruct(const A & a, const B & b) : _1(a), _2(b) { }
    double operator()(double x, double y) const {
      return _1(_2(x, y));
    }
  private:
    A _1; 
    B _2;
  };

  template<typename A, typename B>
  struct Composition {
    typedef CompositionStruct<A, B> type;
    static type compose(const A& a, const B b) { return type(a, b); }
  };

  template<typename A, typename B>
  inline typename funct::Composition<A, B>::type compose(const A& a, const B& b) {
    return funct::Composition<A, B>::compose(a, b);
  }

}


#endif
