#ifndef AnalysisDataFormats_TopObjects_printParticle_h
#define AnalysisDataFormats_TopObjects_printParticle_h
// -*- C++ -*-
//
// Package:     AnalysisDataFormats/TopObjects
// Class  :     printParticle
//
/**\function ttevent::printParticle printParticle.h "AnalysisDataFormats/TopObjects/interface/printParticle.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 16 Oct 2020 13:22:44 GMT
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations

namespace ttevent {
  /// print pt, eta, phi, mass of a given candidate into an existing LogInfo
  void printParticle(edm::LogInfo& log, const char* name, const reco::Candidate* cand);
}  // namespace ttevent
#endif
