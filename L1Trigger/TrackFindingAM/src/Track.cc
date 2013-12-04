#include "../interface/Track.h"

Track::Track(){
  curve = 0;
  d0    = 0;
  phi0  = 0;
  eta0  = 0;
  z0    = 0;
}

Track::Track(double c, double d, double p, double p_a, double p_b){
  curve = c;
  d0    = d;
  phi0  = p;
  eta0  = p_a;
  z0    = p_b;
}

Track::Track(const Track& ref){
  curve = ref.curve;
  d0    = ref.d0;
  phi0  = ref.phi0;
  eta0  = ref.eta0;
  z0    = ref.z0;
}

void Track::setCurve(double c){
  curve=c;
}

void Track::setD0(double d){
  d0=d;
}

void Track::setPhi0(double p){
  phi0=p;
}

void Track::setEta0(double p_a){
  eta0=p_a;
}

void Track::setZ0(double p_b){
  z0=p_b;
}

double Track::getCurve(){
  return curve;
}

double Track::getD0(){
  return d0;
}

double Track::getPhi0(){
  return phi0;
}

double Track::getEta0(){
  return eta0;
}

double Track::getZ0(){
  return z0;
}
