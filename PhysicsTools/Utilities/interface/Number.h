#ifndef PhysicsTools_Utilities_Number_h
#define PhysicsTools_Utilities_Number_h

namespace funct {

  struct Number {
    Number(double value) : value_(value) { }
    double operator()(double x) const { return value_; }
  private:
    double value_;
  };

}

#endif
