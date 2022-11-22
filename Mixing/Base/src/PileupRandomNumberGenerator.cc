// -*- C++ -*-
//
// Package:     Mixing/Base
// Class  :     PileupRandomNumberGenerator
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 09 Nov 2022 17:53:24 GMT
//

// system include files

// user include files
#include "PileupRandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/CurrentModuleOnThread.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "CLHEP/Random/engineIDulong.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RanecuEngine.h"
#include "CLHEP/Random/MixMaxRng.h"
#include "IOMC/RandomEngine/interface/TRandomAdaptor.h"

CLHEP::HepRandomEngine& PileupRandomNumberGenerator::getEngine(edm::StreamID const&) {
  return *m_modulesToEngines[findPresentModule()];
}

CLHEP::HepRandomEngine& PileupRandomNumberGenerator::getEngine(edm::LuminosityBlockIndex const&) {
  return *m_modulesToEngines[findPresentModule()];
}

std::unique_ptr<CLHEP::HepRandomEngine> PileupRandomNumberGenerator::cloneEngine(edm::LuminosityBlockIndex const&) {
  auto& engine = m_modulesToEngines[findPresentModule()];
  return cloneEngine(*engine);
}

void PileupRandomNumberGenerator::setSeed(uint32_t iSeed) { m_seed = iSeed; }

void PileupRandomNumberGenerator::setEngine(CLHEP::HepRandomEngine const& iEngine) {
  for (auto& v : m_modulesToEngines) {
    v.second = cloneEngine(iEngine);
  }
}

void PileupRandomNumberGenerator::preBeginLumi(edm::LuminosityBlock const& lumi) {}
void PileupRandomNumberGenerator::postEventRead(edm::Event const& event) {}

void PileupRandomNumberGenerator::setLumiCache(edm::LuminosityBlockIndex,
                                               std::vector<RandomEngineState> const& iStates) {}
void PileupRandomNumberGenerator::setEventCache(edm::StreamID, std::vector<RandomEngineState> const& iStates) {}

std::vector<RandomEngineState> const& PileupRandomNumberGenerator::getEventCache(edm::StreamID const&) const {
  static const std::vector<RandomEngineState> s_dummy;
  return s_dummy;
}
std::vector<RandomEngineState> const& PileupRandomNumberGenerator::getLumiCache(edm::LuminosityBlockIndex const&) const {
  static const std::vector<RandomEngineState> s_dummy;
  return s_dummy;
}

void PileupRandomNumberGenerator::consumes(edm::ConsumesCollector&& iC) const {}

void PileupRandomNumberGenerator::print(std::ostream& os) const {}

std::unique_ptr<CLHEP::HepRandomEngine> PileupRandomNumberGenerator::cloneEngine(
    CLHEP::HepRandomEngine const& existingEngine) {
  std::vector<unsigned long> stateL = existingEngine.put();
  long seedL = existingEngine.getSeed();
  std::unique_ptr<CLHEP::HepRandomEngine> newEngine;
  if (stateL[0] == CLHEP::engineIDulong<CLHEP::HepJamesRandom>()) {
    newEngine = std::make_unique<CLHEP::HepJamesRandom>(seedL);
  } else if (stateL[0] == CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {
    newEngine = std::make_unique<CLHEP::RanecuEngine>();
  } else if (stateL[0] == CLHEP::engineIDulong<CLHEP::MixMaxRng>()) {
    newEngine = std::make_unique<CLHEP::MixMaxRng>(seedL);
  } else if (stateL[0] == CLHEP::engineIDulong<edm::TRandomAdaptor>()) {
    newEngine = std::make_unique<edm::TRandomAdaptor>(seedL);
  } else {
    // Sanity check, it should not be possible for this to happen.
    throw edm::Exception(edm::errors::Unknown)
        << "The RandomNumberGeneratorService is trying to clone unknown engine type\n";
  }
  newEngine->get(stateL);
  return newEngine;
}

const std::string& PileupRandomNumberGenerator::findPresentModule() const {
  edm::ModuleCallingContext const* mcc = edm::CurrentModuleOnThread::getCurrentModuleOnThread();
  if (mcc == nullptr) {
    throw edm::Exception(edm::errors::LogicError)
        << "PileupRandomNumberGenerator::getEngine()\n"
           "Requested a random number engine from the RandomNumberGeneratorService\n"
           "from an unallowed transition. ModuleCallingContext is null\n";
  }
  return mcc->moduleDescription()->moduleLabel();
}
