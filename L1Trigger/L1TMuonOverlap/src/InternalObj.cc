#include "L1Trigger/L1TMuonOverlap/interface/InternalObj.h"

#include <bitset>

std::ostream & operator<< (std::ostream &out, const InternalObj &o){
  out<<"InternalObj: ";
  out <<" pt: "<<o.pt
      <<", eta: "<<o.eta*2.61/240
      <<", phi: "<<o.phi
      <<", charge: "<<o.charge
      <<", q: "<<o.q
      <<" hits: "<<std::bitset<18>(o.hits).to_string()
      <<", bx: "<<o.bx
      <<", disc: "<<o.disc<<" refLayer: "<<o.refLayer;
  
  return out;
}

