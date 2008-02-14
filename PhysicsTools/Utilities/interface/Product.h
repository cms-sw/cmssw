#ifndef PhysicsTools_Utilities_Product_h
#define PhysicsTools_Utilities_Product_h

namespace function {
  template<typename A, typename B>
  class Product { 
  public:
    enum { arguments = 1 };
    enum { parameters = A::parameters + B::parameters }; 
    Product(const A & a, const B & b) : a_(a), b_(b) { }
    double operator()(double x) const {
      return a_(x) * b_(x);
    }
  private:
    A a_; 
    B b_;
  };
}

template<typename A, typename B>
function::Product<A, B> operator*(const A& a, const B& b) {
  return function::Product<A, B>(a, b);
}

#endif
