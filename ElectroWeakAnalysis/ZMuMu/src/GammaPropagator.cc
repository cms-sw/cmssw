#include "ElectroWeakAnalysis/ZMuMu/interface/GammaPropagator.h"

GammaPropagator::GammaPropagator(){
  setParameters();
}

void GammaPropagator::setParameters(){
  g_ = 1;
}

double GammaPropagator::operator()(double x) const{
  double s = x*x;
  double lineShape = 0;
  lineShape =  g_/s;
  return lineShape;
}
