#include "FWCore/Services/interface/ExternalRandomNumberGeneratorService.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RanecuEngine.h"
#include "CLHEP/Random/MixMaxRng.h"

//#define CATCH_CONFIG_MAIN
#include "catch.hpp"

namespace {
  void test(CLHEP::HepRandomEngine& iRand, CLHEP::HepRandomEngine& iEngine) {
    REQUIRE(iRand.flat() == iEngine.flat());
    REQUIRE(iRand.flat() == iEngine.flat());
    REQUIRE(iRand.flat() == iEngine.flat());
    REQUIRE(iRand.flat() == iEngine.flat());
  }
}  // namespace

TEST_CASE("Test ExternalRandomNumberGeneratorService", "[externalrandomnumbergeneratorservice]") {
  SECTION("JamesRandom") {
    edm::ExternalRandomNumberGeneratorService service;
    CLHEP::HepJamesRandom rand(12345);

    service.setState(rand.put(), rand.getSeed());
    test(rand, service.getEngine(edm::StreamID::invalidStreamID()));

    //advance the one to see how it works
    rand.flat();
    service.setState(rand.put(), rand.getSeed());
    test(rand, service.getEngine(edm::StreamID::invalidStreamID()));
  }

  SECTION("RanecuEngine") {
    edm::ExternalRandomNumberGeneratorService service;
    CLHEP::RanecuEngine rand(12345);

    service.setState(rand.put(), rand.getSeed());
    test(rand, service.getEngine(edm::StreamID::invalidStreamID()));

    //advance the one to see how it works
    rand.flat();
    service.setState(rand.put(), rand.getSeed());
    test(rand, service.getEngine(edm::StreamID::invalidStreamID()));
  }

  SECTION("MixMaxRng") {
    edm::ExternalRandomNumberGeneratorService service;
    CLHEP::MixMaxRng rand(12345);

    service.setState(rand.put(), rand.getSeed());
    test(rand, service.getEngine(edm::StreamID::invalidStreamID()));

    //advance the one to see how it works
    rand.flat();
    service.setState(rand.put(), rand.getSeed());
    test(rand, service.getEngine(edm::StreamID::invalidStreamID()));
  }
}
