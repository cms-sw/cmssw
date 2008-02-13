#ifndef PhysicsTools_Utilities_GammaPropagator_h
#define PhysicsTools_Utilities_GammaPropagator_h

struct GammaPropagator {
  enum { arguments = 1 };
  enum { parameters = 0 };
  GammaPropagator() {}
  void setParameters() {}
  double operator()(double mass) const { 
    double s = mass*mass;
    return 1./s;
  }
};

#endif
