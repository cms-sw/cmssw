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
  out << "finalMuon"
      << " pt " << std::setw(8) << finalMuon.ptGev << " GeV"
      << " eta " << std::setw(8) << finalMuon.etaRad
      << " phi " << std::setw(8) << finalMuon.phiRad
      << " sign   " << std::setw(2) << finalMuon.sign
      << " ptGmt " << std::setw(8) << finalMuon.getPtGmt()
      << " etaGmt " << std::setw(8) << finalMuon.getEtaGmt()
      << " phiGmt " << std::setw(8) << finalMuon.getPhiGmt()
      //<< " hwPhi " << muonCand->hwPhi()
      << " hwQual " << finalMuon.getQuality() << " processor " << finalMuon.getProcessor();
  return out;
}


