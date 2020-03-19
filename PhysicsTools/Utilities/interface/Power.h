#ifndef PhysicsTools_Utilities_Power_h
#define PhysicsTools_Utilities_Power_h
#include <boost/static_assert.hpp>
#include <cmath>

namespace funct {
  template <typename A, typename B>
  struct PowerStruct {
    PowerStruct(const A& a, const B& b) : _1(a), _2(b) {}
    double operator()() const { return std::pow(_1(), _2()); }
    operator double() const { return std::pow(_1(), _2()); }
    double operator()(double x) const { return std::pow(_1(x), _2(x)); }
    double operator()(double x, double y) const { return std::pow(_1(x, y), _2(x, y)); }
    A _1;
    B _2;
  };

  template <typename A, typename B>
  struct Power {
    typedef PowerStruct<A, B> type;
    static type combine(const A& a, const B& b) { return type(a, b); }
  };

  template <typename A, typename B>
  inline typename Power<A, B>::type operator^(const A& a, const B& b) {
    return Power<A, B>::combine(a, b);
  }

  template <typename A, typename B>
  inline typename Power<A, B>::type pow(const A& a, const B& b) {
    return Power<A, B>::combine(a, b);
  }

}  // namespace funct

#endif
