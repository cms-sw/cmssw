#ifndef PhysicsTools_Utilities_Minus_h
#define PhysicsTools_Utilities_Minus_h

namespace funct {

  template<typename A, unsigned int args = A::arguments>
  class MinusStruct { 
  public:
    static const unsigned int arguments = args;
  };

  template<typename A>
  class MinusStruct<A, 0> { 
  public:
    static const unsigned int arguments = 0;
    MinusStruct(const A & a) : a_(a) { }
    double operator()() const {
      return - a_();
    }
    operator double() const {
      return - a_();
    }
  private:
    A a_; 
  };

  template<typename A>
  class MinusStruct<A, 1> { 
  public:
    static const unsigned int arguments = 1;
    MinusStruct(const A & a) : a_(a) { }
    double operator()(double x) const {
      return - a_(x);
    }
  private:
    A a_; 
  };

  template<typename A>
  class MinusStruct<A, 2> { 
  public:
    static const unsigned int arguments = 2;
    MinusStruct(const A & a) : a_(a) { }
    double operator()(double x, double y) const {
      return - a_(x, y);
    }
  private:
    A a_; 
  };

  template<typename A>
  struct Minus {
    typedef MinusStruct<A> type;
    static type operate(const A& a) {
      return type(a);
    }
  };
}

#endif
