#ifndef PhysicsTools_Utilities_Minus_h
#define PhysicsTools_Utilities_Minus_h

namespace function {
  template<typename A, unsigned int args = A::arguments>
  class Minus { 
  public:
    static const unsigned int arguments = args;
  };

  template<typename A>
  class Minus<A, 0> { 
  public:
    static const unsigned int arguments = 0;
    Minus(const A & a) : a_(a) { }
    double operator()() const {
      return - a_();
    }
  private:
    A a_; 
  };

  template<typename A>
  class Minus<A, 1> { 
  public:
    static const unsigned int arguments = 1;
    Minus(const A & a) : a_(a) { }
    double operator()(double x) const {
      return - a_(x);
    }
  private:
    A a_; 
  };

  template<typename A>
  class Minus<A, 2> { 
  public:
    static const unsigned int arguments = 2;
    Minus(const A & a) : a_(a) { }
    double operator()(double x, double y) const {
      return - a_(x, y);
    }
  private:
    A a_; 
  };
}
template<typename A>
function::Minus<A> operator-(const A& a) {
  return function::Minus<A>(a);
}

#endif
