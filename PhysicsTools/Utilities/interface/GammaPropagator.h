#ifndef PhysicsTools_Utilities_GammaPropagator_h
#define PhysicsTools_Utilities_GammaPropagator_h

namespace function {

  struct GammaPropagator {
    enum { arguments = 1 };
    enum { parameters = 0 };
    GammaPropagator() {}
    double operator()(double mass) const { 
      if(mass <= 0) return 0;
      return 1./(mass*mass);
    }
  };

}

#endif
