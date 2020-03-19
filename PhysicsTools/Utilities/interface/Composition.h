#ifndef PhysicsTools_Utilities_Composition_h
#define PhysicsTools_Utilities_Composition_h
#include <boost/static_assert.hpp>

namespace funct {
  template <typename A, typename B>
  struct CompositionStruct {
    CompositionStruct(const A& a, const B& b) : _1(a), _2(b) {}
    double operator()() const { return _1(_2()); }
    operator double() const { return _1(_2()); }
    double operator()(double x) const { return _1(_2(x)); }
    double operator()(double x, double y) const { return _1(_2(x, y)); }
    A _1;
    B _2;
  };

  template <typename A, typename B>
  struct Composition {
    typedef CompositionStruct<A, B> type;
    static type compose(const A& a, const B& b) { return type(a, b); }
  };

  template <typename A, typename B>
  inline typename funct::Composition<A, B>::type compose(const A& a, const B& b) {
    return funct::Composition<A, B>::compose(a, b);
  }

}  // namespace funct

#endif
