#ifndef PhysicsTools_Utilities_Numerical_h
#define PhysicsTools_Utilities_Numerical_h

namespace funct {

  template<int n> struct Numerical {
    static const int value;
    double operator()() const { return n; }
  };

  template<int n> 
  const int Numerical<n>::value = m;

  template<int n> const Numerical<n>& num()
  { static Numerical<n> c; return c; }

#define num(n) num<n>()

}

#endif
