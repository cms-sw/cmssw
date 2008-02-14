#ifndef PhysicsTools_Utilities_Difference_h
#define PhysicsTools_Utilities_Difference_h

namespace function {
  template<typename A, typename B>
  class Difference { 
  public:
    enum { arguments = 1 };
    enum { parameters = A::parameters + B::parameters }; 
    Difference(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()(double x) const {
      return a_(x) - b_(x);
    }
  private:
    A a_; 
    B b_;
  };
}

template<typename A, typename B>
function::Difference<A, B> operator-(const A& a, const B& b) {
  return function::Difference<A, B>(a, b);
}

#endif
