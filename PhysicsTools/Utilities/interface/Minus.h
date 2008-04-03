#ifndef PhysicsTools_Utilities_Minus_h
#define PhysicsTools_Utilities_Minus_h

namespace funct {

  template<typename A, unsigned int args = A::arguments>
  struct MinusStruct { 
    static const unsigned int arguments = args;
  };

  template<typename A>
  struct MinusStruct<A, 0> { 
    static const unsigned int arguments = 0;
    MinusStruct() : _() { }
    MinusStruct(const A & a) : _(a) { }
    double operator()() const {
      return - _();
    }
    operator double() const {
      return - _();
    }
    A _; 
  };

  template<typename A>
  struct MinusStruct<A, 1> { 
    static const unsigned int arguments = 1;
    MinusStruct() : _() { }
    MinusStruct(const A & a) : _(a) { }
    double operator()(double x) const {
      return - _(x);
    }
    A _; 
  };

  template<typename A>
  struct MinusStruct<A, 2> { 
    static const unsigned int arguments = 2;
    MinusStruct() : _() { }
    MinusStruct(const A & a) : _(a) { }
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
