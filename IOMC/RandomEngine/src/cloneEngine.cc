// -*- C++ -*-
//
// Package:     IOMC/RandomEngine
// Class  :     cloneEngine
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 02 Dec 2022 19:34:37 GMT
//

// system include files

// user include files
#include "IOMC/RandomEngine/interface/cloneEngine.h"
#include "IOMC/RandomEngine/interface/TRandomAdaptor.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include "CLHEP/Random/engineIDulong.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RanecuEngine.h"
#include "CLHEP/Random/MixMaxRng.h"

namespace edm {
  std::unique_ptr<CLHEP::HepRandomEngine> cloneEngine(CLHEP::HepRandomEngine const& existingEngine) {
    std::vector<unsigned long> stateL = existingEngine.put();
    long seedL = existingEngine.getSeed();
    std::unique_ptr<CLHEP::HepRandomEngine> newEngine;
    if (stateL[0] == CLHEP::engineIDulong<CLHEP::HepJamesRandom>()) {
      newEngine = std::make_unique<CLHEP::HepJamesRandom>(seedL);
    } else if (stateL[0] == CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {
      newEngine = std::make_unique<CLHEP::RanecuEngine>();
    } else if (stateL[0] == CLHEP::engineIDulong<CLHEP::MixMaxRng>()) {
      newEngine = std::make_unique<CLHEP::MixMaxRng>(seedL);
    } else if (stateL[0] == CLHEP::engineIDulong<TRandomAdaptor>()) {
      newEngine = std::make_unique<TRandomAdaptor>(seedL);
    } else {
      // Sanity check, it should not be possible for this to happen.
      throw Exception(errors::Unknown) << "The RandomNumberGeneratorService is trying to clone unknown engine type\n";
    }
    newEngine->get(stateL);
    return newEngine;
  }
};  // namespace edm
