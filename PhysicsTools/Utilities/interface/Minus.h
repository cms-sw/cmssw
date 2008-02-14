#ifndef PhysicsTools_Utilities_Minus_h
#define PhysicsTools_Utilities_Minus_h

namespace function {
  template<typename A>
  class Minus { 
  public:
    enum { arguments = 1 };
    enum { parameters = A::parameters }; 
    Minus(const A & a) : a_(a) { }
    double operator()(double x) const {
      return - a_(x);
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
