#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <iostream>

bool AlgoMuon::isValid() const {
  return getPt() > 0;  //should this realy be pt or quality ?? FIXME
}

bool AlgoMuon::operator<(const AlgoMuon &o) const {
  if (this->getQ() > o.getQ())
    return false;
  else if (this->getQ() == o.getQ() && this->getDisc() > o.getDisc())
    return false;
  else if (getQ() == o.getQ() && getDisc() == o.getDisc() && getPatternNumber() > o.getPatternNumber())
    return false;
  else if (getQ() == o.getQ() && getDisc() == o.getDisc() && getPatternNumber() == o.getPatternNumber() &&
           getRefHitNumber() < o.getRefHitNumber())
    return false;
  else
    return true;
}

std::ostream &operator<<(std::ostream &out, const AlgoMuon &o) {
  out << "AlgoMuon: ";
  out << " pt: " << o.getPt() << ", phi: " << o.getPhi() << ", eta: " << o.getEtaHw()
      << ", hits: " << std::bitset<18>(o.getFiredLayerBits()).to_string() << ", q: " << o.getQ()
      << ", bx: " << o.getBx() << ", charge: " << o.getCharge() << ", disc: " << o.getDisc()
      << " refLayer: " << o.getRefLayer() << " m_patNumb: " << o.getPatternNumber();

  return out;
}
