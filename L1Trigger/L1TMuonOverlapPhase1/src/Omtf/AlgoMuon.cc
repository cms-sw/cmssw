#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <iostream>

std::ostream &operator<<(std::ostream &out, const AlgoMuon &o) {
  out << "AlgoMuon: ";
  out << " pt: " << o.getPtConstr() << " upt: " << o.getPtUnconstr() << ", phi: " << o.getPhi()
      << ", eta: " << o.getEtaHw() << ", hits: " << std::bitset<18>(o.getFiredLayerBits()).to_string()
      << ", q: " << o.getQ() << ", bx: " << o.getBx() << ", charge: " << o.getChargeConstr()
      << ", disc: " << o.getDisc() << " refLayer: " << o.getRefLayer() << " m_patNumb: " << o.getPatternNumConstr();

  return out;
}

unsigned int AlgoMuon::getPatternNum() const {
  if (goldenPaternUnconstr == nullptr)
    return goldenPaternConstr->key().theNumber;

  return (gpResultUnconstr.getPdfSumUnconstr() > gpResultConstr.getPdfSum() ? goldenPaternUnconstr->key().theNumber
                                                                            : goldenPaternConstr->key().theNumber);
}
