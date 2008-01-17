#ifndef PhysicsTools_Utilities_GammaPropagator_h
#define PhysicsTools_Utilities_GammaPropagator_h

class GammaPropagator {
 public:
  GammaPropagator();
  void setParameters();
  double operator()(double mass) const;
};

#endif
