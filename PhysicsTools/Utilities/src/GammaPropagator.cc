#include "ElectroWeakAnalysis/ZMuMu/interface/GammaPropagator.h"

GammaPropagator::GammaPropagator() {
}

void GammaPropagator::setParameters(){
}

double GammaPropagator::operator()(double mass) const{
  double s = mass*mass;
  return 1./s;
}
