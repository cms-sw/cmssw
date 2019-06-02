#ifndef PhysicsTools_Utilities_Ratio_h
#define PhysicsTools_Utilities_Ratio_h
#include <boost/static_assert.hpp>

namespace funct {
  template <typename A, typename B>
  struct RatioStruct {
    RatioStruct(const A& a, const B& b) : _1(a), _2(b) {}
    double operator()() const { return _1() / _2(); }
    operator double() const { return _1() / _2(); }
    double operator()(double x) const { return _1(x) / _2(x); }
    double operator()(double x, double y) const { return _1(x, y) / _2(x, y); }
    A _1;
    B _2;
  };

  template <typename A, typename B>
  struct Ratio {
    typedef RatioStruct<A, B> type;
    static type combine(const A& a, const B& b) { return type(a, b); }
  };

  template <typename A, typename B>
  inline typename Ratio<A, B>::type operator/(const A& a, const B& b) {
    return Ratio<A, B>::combine(a, b);
  }

}  // namespace funct

#endif
