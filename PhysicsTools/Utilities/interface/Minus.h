#ifndef PhysicsTools_Utilities_Minus_h
#define PhysicsTools_Utilities_Minus_h

namespace funct {

  template<typename A>
  struct MinusStruct { 
    MinusStruct() : _() { }
    MinusStruct(const A & a) : _(a) { }
    operator double() const {
      return - _();
    }
    double operator()() const {
      return - _();
    }
    double operator()(double x) const {
      return - _(x);
    }
    double operator()(double x, double y) const {
      return - _(x, y);
    }
    A _; 
  };

  template<typename A>
  struct Minus {
    typedef MinusStruct<A> type;
    static type operate(const A& a) {
      return type(a);
    }
  };

  template<typename A>
  inline typename Minus<A>::type operator-(const A& a) {
    return Minus<A>::operate(a);
  }
}

#endif
