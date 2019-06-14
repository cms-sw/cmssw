#ifndef PhysicsTools_Utilities_Numerical_h
#define PhysicsTools_Utilities_Numerical_h
#include <cassert>
namespace funct {

  template <int n>
  struct Numerical {
    Numerical() {}
    Numerical(int m) { assert(m == n); }
    static const int value = n;
    double operator()() const { return n; }
    operator double() const { return n; }
    double operator()(double) const { return n; }
    double operator()(double, double) const { return n; }
  };

  template <int n>
  const Numerical<n>& num() {
    static Numerical<n> c;
    return c;
  }

}  // namespace funct

#endif
