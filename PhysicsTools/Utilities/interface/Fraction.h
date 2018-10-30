#ifndef PhysicsTools_Utilities_Fraction_h
#define PhysicsTools_Utilities_Fraction_h

#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/Operations.h"
#include <boost/math/common_factor.hpp>
#include <boost/static_assert.hpp>

namespace funct {

  template<int n, int m>
  struct FractionStruct { 
    static_assert(m != 0);
    static const int numerator = n, denominator = m;
    double operator()() const { return double(n) / double (m); }
    operator double() const { return double(n) / double (m); }
    double operator()(double) const { return double(n) / double (m); }
    double operator()(double, double) const { return double(n) / double (m); }
  };
  
  template<int n, int m, unsigned gcd = boost::math::static_gcd<n, m>::value,
    int num = n / gcd, int den = m / gcd>
  struct PositiveFraction {
    typedef FractionStruct<num, den> type;
  };
  
  template<int n, int m, unsigned gcd, int num>
  struct PositiveFraction<n, m , gcd, num, 1> {
    typedef Numerical<num> type;
  };
  
  template<int n, int m, bool pn = (n >= 0), bool pm = (m >= 0)>
  struct Fraction { 
    typedef typename PositiveFraction<n, m>::type type; 
  };
  
  template<int n, int m> 
  const typename Fraction<n, m>::type& fract() { 
    static typename Fraction<n, m>::type c; 
    return c; 
  }

  template<int n, int m>
  struct Ratio<Numerical<n>, Numerical<m> > { 
    typedef typename Fraction<n, m>::type type; 
    inline static type combine(const Numerical<n>&, const Numerical<m>&) { 
      return fract<n, m>(); 
    }
  };

  template<int n, int m>
  struct Fraction<n, m, false, true> { 
    typedef typename Minus<typename PositiveFraction<-n, m>::type>::type type; 
  };

  template<int n, int m>
  struct Fraction<n, m, true, false> { 
    typedef typename Minus<typename PositiveFraction<n, -m>::type>::type type; 
  };

  template<int n, int m>
  struct Fraction<n, m, false, false> { 
    typedef typename Minus<typename PositiveFraction<-n, -m>::type>::type type; 
  };

  template<int a, int b, int c>
  struct Product<Numerical<a>, FractionStruct<b, c> > { 
    typedef typename Fraction<a * b, c>::type type;
    inline static type combine(const Numerical<a>&, const FractionStruct<b, c>&) { 
      return fract<a * b, c>(); 
    }
  };

  template<int a, int b, int c>
  struct Product<FractionStruct<b, c>, Numerical<a> > { 
    typedef typename Fraction<a * b, c>::type type;
    inline static type combine(const FractionStruct<b, c>&, const Numerical<a>&) { 
      return fract<a * b, c>(); 
    }
  };
  
  template<int a, int b, int c>
  struct Ratio<Numerical<a>, FractionStruct<b, c> > { 
    typedef typename Fraction<a * c, b>::type type;
    inline static type combine(const Numerical<a>&, const FractionStruct<b, c>&) { 
      return fract<a * c, b>(); 
    }
  };

  template<int a, int b, int c>
  struct Sum<Numerical<a>, FractionStruct<b, c> > { 
    typedef typename Fraction<a * c + b, b>::type type;
    inline static type combine(const Numerical<a>&, const FractionStruct<b, c>&) {
      return fract<a * c + b, b>(); 
    }
  };

  template<int a, int b, int c>
  struct Difference<Numerical<a>, FractionStruct<b, c> > { 
    typedef typename Fraction<a * c - b, b>::type type;
    inline static type combine(const Numerical<a>&, const FractionStruct<b, c>&) { 
      return fract<a * c - b, b>(); 
    }
  };
  
  template<int a, int b, int c>
  struct Sum<FractionStruct<b, c>, Numerical<a> > { 
    typedef typename Fraction<a * c + b, b>::type type;
    inline static type combine(const FractionStruct<b, c>&, const Numerical<a>&) { 
      return fract<a * c + b, b>(); 
    }
  };

  template<int a, int b, int c>
  struct Ratio<FractionStruct<b, c>, Numerical<a> > { 
    typedef typename Fraction<b, a * c>::type type;
    inline static type combine(const FractionStruct<b, c>&, const Numerical<a>&) { 
      return fract<b, a * c>(); 
    }
  };

  template<int a, int b, int c, int d>
  struct Sum<FractionStruct<a, b>, FractionStruct<c, d> > { 
    typedef typename Fraction<a * d + c * b, b * d>::type type; 
    inline static type combine(const FractionStruct<a, b>&, const FractionStruct<c, d>&) { 
      return fract<a * d + c * b, b * d>(); 
    }
  };
  
  template<int a, int b, int c, int d>
  struct Difference<FractionStruct<a, b>, FractionStruct<c, d> > { 
    typedef typename Fraction<a * d - c * b, b * d>::type type; 
    inline static type combine(const FractionStruct<a, b>&, const FractionStruct<c, d>&) { 
      return fract<a * d - c * b, b * d>(); 
    }
  };

  template<int a, int b, int c, int d>
  struct Product<FractionStruct<a, b>, FractionStruct<c, d> > { 
    typedef typename Fraction<a * c, b * d>::type type; 
    inline static type combine(const FractionStruct<a, b>&, const FractionStruct<c, d>&) { 
      return fract<a * c, b * d>(); 
    }
  };

  template<int a, int b, int c, int d>
  struct Ratio<FractionStruct<a, b>, FractionStruct<c, d> > { 
    typedef typename Fraction<a * d, b * c>::type type; 
    inline static type combine(const FractionStruct<a, b>&, const FractionStruct<c, d>& ) { 
      return fract<a * d, b * c>(); 
    }
  };

}

#endif
