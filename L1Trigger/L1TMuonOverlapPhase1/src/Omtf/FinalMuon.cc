/*
 * FinalMuon.cc
 *
 *  Created on: Nov 10, 2025
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/FinalMuon.h"

#include <iostream>
#include <iomanip>

std::ostream &operator<<(std::ostream &out, const FinalMuon &finalMuon) {
  out << "finalMuon";
  out << " pt " << std::setw(8) << finalMuon.ptGev << " GeV";
  out << " phi " << std::setw(8) << finalMuon.phiRad;
  out << " sign   " << std::setw(2) << finalMuon.sign;
  out << " eta " << std::setw(8) << finalMuon.etaRad;
  out << " ptGmt " << std::setw(8) << finalMuon.getPtGmt();
  out << " etaGmt " << std::setw(8) << finalMuon.getEtaGmt();
  out << " phiGmt " << std::setw(8) << finalMuon.getPhiGmt();
  //<< " hwPhi " << muonCand->hwPhi();
  out << " hwQual " << finalMuon.getQuality() << " processor " << finalMuon.getProcessor();
  return out;
}
