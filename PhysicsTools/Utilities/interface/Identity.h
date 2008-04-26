#ifndef PhysicsTools_Utilities_Identity_h
#define PhysicsTools_Utilities_Identity_h

namespace function {
  
  struct Identity {
    static const unsigned int arguments = 1;
    Identity() { }
    double operator()(double x) const {
      return x;
    }
  };

}

#endif
