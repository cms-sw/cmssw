#ifndef PhysicsTools_Utilities_GammaPropagator_h
#define PhysicsTools_Utilities_GammaPropagator_h

namespace funct {

  struct GammaPropagator {
    GammaPropagator() {}
    double operator()(double mass) const { 
      if(mass <= 0) return 0;
      return 1./(mass*mass);
    }
  };

}

#endif
