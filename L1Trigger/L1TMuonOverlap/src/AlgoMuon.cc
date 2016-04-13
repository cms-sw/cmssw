#include "L1Trigger/L1TMuonOverlap/interface/AlgoMuon.h"

#include <bitset> 
#include <iostream>

bool AlgoMuon::isValid() const{
 return m_q >= 0;
}

bool AlgoMuon::operator< (const AlgoMuon & o) const{ 
  if(this->getQ() > o.getQ()) return false;
  else if(this->getQ()==o.getQ() && this->getDisc() > o.getDisc()) return false;
  else return true;
}

std::ostream & operator<< (std::ostream &out, const AlgoMuon &o){
  out <<"AlgoMuon: ";
  out << " pt: "   << o.getPt()
      << ", phi: " << o.getPhi()
      << ", eta: " << o.getEta()*2.61/240
      << ", hits: " << std::bitset<18>(o.getHits()).to_string()
      << ", q: "   << o.getQ()
      << ", bx: "  << o.getBx()
      << ", charge: "<< o.getCharge()
      << ", disc: "  << o.getDisc() << " refLayer: " << o.getRefLayer();
  
  return out;
}
