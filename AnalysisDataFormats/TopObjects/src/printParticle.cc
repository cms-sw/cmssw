// -*- C++ -*-
//
// Package:     AnalysisDataFormats/TopObjects
// Class  :     printParticle
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 16 Oct 2020 13:25:54 GMT
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/src/printParticle.h"

namespace ttevent {
  // print pt, eta, phi, mass of a given candidate into an existing LogInfo
  void printParticle(edm::LogInfo& log, const char* name, const reco::Candidate* cand) {
    if (!cand) {
      log << std::setw(15) << name << ": not available!\n";
      return;
    }
    log << std::setprecision(3) << setiosflags(std::ios::fixed | std::ios::showpoint);
    log << std::setw(15) << name << ": " << std::setw(7) << cand->pt() << "; " << std::setw(7) << cand->eta() << "; "
        << std::setw(7) << cand->phi() << "; " << resetiosflags(std::ios::fixed | std::ios::showpoint)
        << setiosflags(std::ios::scientific) << std::setw(10) << cand->mass() << "\n";
    log << resetiosflags(std::ios::scientific);
  }

}  // namespace ttevent
