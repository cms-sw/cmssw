#include "FWCore/Services/interface/ExternalRandomNumberGeneratorService.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "CLHEP/Random/engineIDulong.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RanecuEngine.h"
#include "CLHEP/Random/MixMaxRng.h"

#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"

using namespace edm;

namespace {
  const std::vector<RandomEngineState> s_dummyStates;
}

ExternalRandomNumberGeneratorService::ExternalRandomNumberGeneratorService() {}

void ExternalRandomNumberGeneratorService::setState(std::vector<unsigned long> const& iState, long iSeed) {
  if (not engine_) {
    engine_ = createFromState(iState, iSeed);
  } else {
    engine_->get(iState);
  }
}

std::vector<unsigned long> ExternalRandomNumberGeneratorService::getState() const { return engine_->put(); }

CLHEP::HepRandomEngine& ExternalRandomNumberGeneratorService::getEngine(StreamID const&) { return *engine_; }
CLHEP::HepRandomEngine& ExternalRandomNumberGeneratorService::getEngine(LuminosityBlockIndex const&) {
  return *engine_;
}

std::unique_ptr<CLHEP::HepRandomEngine> ExternalRandomNumberGeneratorService::cloneEngine(LuminosityBlockIndex const&) {
  std::vector<unsigned long> stateL = engine_->put();

  long seedL = engine_->getSeed();
  return createFromState(stateL, seedL);
}

std::unique_ptr<CLHEP::HepRandomEngine> ExternalRandomNumberGeneratorService::createFromState(
    std::vector<unsigned long> const& stateL, long seedL) const {
  std::unique_ptr<CLHEP::HepRandomEngine> newEngine;
  if (stateL[0] == CLHEP::engineIDulong<CLHEP::HepJamesRandom>()) {
    newEngine = std::make_unique<CLHEP::HepJamesRandom>(seedL);
  } else if (stateL[0] == CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {
    newEngine = std::make_unique<CLHEP::RanecuEngine>();
  } else if (stateL[0] == CLHEP::engineIDulong<CLHEP::MixMaxRng>()) {
    newEngine = std::make_unique<CLHEP::MixMaxRng>(seedL);
    //} else if (stateL[0] == CLHEP::engineIDulong<TRandomAdaptor>()) {
    //  newEngine = std::make_unique<TRandomAdaptor>(seedL);
  } else {
    // Sanity check, it should not be possible for this to happen.
    throw Exception(errors::Unknown)
        << "The ExternalRandomNumberGeneratorService is trying to clone unknown engine type\n";
  }
  if (stateL[0] != CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {
    newEngine->setSeed(seedL, 0);
  }
  newEngine->get(stateL);
  return newEngine;
}

std::uint32_t ExternalRandomNumberGeneratorService::mySeed() const { return 0; }

void ExternalRandomNumberGeneratorService::preBeginLumi(LuminosityBlock const&) {}

void ExternalRandomNumberGeneratorService::postEventRead(Event const&) {}

void ExternalRandomNumberGeneratorService::setLumiCache(LuminosityBlockIndex,
                                                        std::vector<RandomEngineState> const& iStates) {}
void ExternalRandomNumberGeneratorService::setEventCache(StreamID, std::vector<RandomEngineState> const& iStates) {}

std::vector<RandomEngineState> const& ExternalRandomNumberGeneratorService::getEventCache(StreamID const&) const {
  return s_dummyStates;
}

std::vector<RandomEngineState> const& ExternalRandomNumberGeneratorService::getLumiCache(
    LuminosityBlockIndex const&) const {
  return s_dummyStates;
}

void ExternalRandomNumberGeneratorService::consumes(ConsumesCollector&& iC) const {}

/// For debugging purposes only.
void ExternalRandomNumberGeneratorService::print(std::ostream& os) const {}
