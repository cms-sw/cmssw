#ifndef Mixing_Base_PileupRandomNumberGenerator_h
#define Mixing_Base_PileupRandomNumberGenerator_h
// -*- C++ -*-
//
// Package:     Mixing/Base
// Class  :     PileupRandomNumberGenerator
//
/**\class PileupRandomNumberGenerator PileupRandomNumberGenerator.h "PileupRandomNumberGenerator.h"

 Description: Handle forwarding random numbers to modules within a mixing module

 Usage:
    Internal

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 09 Nov 2022 17:53:22 GMT
//

// system include files
#include <memory>
#include <unordered_map>

// user include files
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

// forward declarations

class PileupRandomNumberGenerator : public edm::RandomNumberGenerator {
public:
  PileupRandomNumberGenerator(std::vector<std::string> const& iModuleLabels);

  void setSeed(uint32_t iSeed);
  void setEngine(CLHEP::HepRandomEngine const&);

  CLHEP::HepRandomEngine& getEngine(edm::StreamID const&) final;

  CLHEP::HepRandomEngine& getEngine(edm::LuminosityBlockIndex const&) final;

  std::unique_ptr<CLHEP::HepRandomEngine> cloneEngine(edm::LuminosityBlockIndex const&) final;

  std::uint32_t mySeed() const final { return m_seed; }

private:
  void preBeginLumi(edm::LuminosityBlock const& lumi) final;
  void postEventRead(edm::Event const& event) final;

  void setLumiCache(edm::LuminosityBlockIndex, std::vector<RandomEngineState> const& iStates) final;
  void setEventCache(edm::StreamID, std::vector<RandomEngineState> const& iStates) final;

  std::vector<RandomEngineState> const& getEventCache(edm::StreamID const&) const final;
  std::vector<RandomEngineState> const& getLumiCache(edm::LuminosityBlockIndex const&) const final;

  void consumes(edm::ConsumesCollector&& iC) const final;

  /// For debugging purposes only.
  void print(std::ostream& os) const final;

  static std::unique_ptr<CLHEP::HepRandomEngine> cloneEngine(CLHEP::HepRandomEngine const& existingEngine);

  const std::string& findPresentModule() const;

  std::unordered_map<std::string, std::unique_ptr<CLHEP::HepRandomEngine>> m_modulesToEngines;
  uint32_t m_seed;
};

#endif
