#ifndef PhysicsTools_Utilities_Identity_h
#define PhysicsTools_Utilities_Identity_h

namespace funct {

  struct Identity {
    Identity() {}
    double operator()(double x) const { return x; }
  };

}  // namespace funct

#endif
