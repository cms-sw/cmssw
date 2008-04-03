#ifndef PhysicsTools_Utilities_Numerical_h
#define PhysicsTools_Utilities_Numerical_h

namespace funct {

  template<int n> struct Numerical {
    static const int value = n;
    double operator()() const { return n; }
    operator double() const { return n; }
  };

  template<int n> const Numerical<n>& num()
  { static Numerical<n> c; return c; }

}

#endif
