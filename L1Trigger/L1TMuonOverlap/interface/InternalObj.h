#ifndef InternalObj_H
#define InternalObj_H

#include <ostream>

struct InternalObj{

  float pt, eta, phi;
  float disc;
  int   bx, q, charge;
  int refLayer;
  int hits;

  InternalObj() : pt(-1.), eta(99.), phi(9999.), disc(-999), bx(0), q(-1), charge(99), refLayer(-1) {}
  InternalObj(float pt, float eta, float phi, 
	      float disc=-999, int bx=0, int q=-1, 
	      int charge=99, int refLayer=-1) :
  pt(pt), eta(eta), phi(phi), disc(disc), bx(bx), q(q), charge(charge), refLayer(refLayer), hits(0) {}

  bool isValid() const { return q >= 0;}

  bool operator< (const InternalObj & o) const { 
    if(q > o.q) return true;
    else if(q==o.q && disc > o.disc) return true;
    else return false;

  }

  friend std::ostream & operator<< (std::ostream &out, const InternalObj &o);

};
#endif
