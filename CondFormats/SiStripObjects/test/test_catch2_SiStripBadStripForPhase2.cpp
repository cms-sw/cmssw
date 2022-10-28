#include "catch.hpp"
#include <iostream>
#include <iomanip>  // std::setw
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

TEST_CASE("SiStripBadStrip testing", "[SiStripBadStrip]") {
  //_____________________________________________________________
  SECTION("Check barrel plotting") {
    SiStripBadStrip testObject;

    static constexpr unsigned short maxStrips = 0x7FF;

    int counter{0};
    for (unsigned short fs = 0; fs <= maxStrips; fs++) {
      for (unsigned short rng = 0; rng <= maxStrips; rng++) {
        auto encoded = testObject.encodePhase2(fs, rng);
        auto decoded = testObject.decodePhase2(encoded);

        if (counter < 10) {
          std::cout << "input: (" << fs << "," << rng << ") | encoded:" << std::setw(10) << encoded << "| decoded : ("
                    << decoded.firstStrip << "," << decoded.range << ")" << std::endl;
        }

        assert(decoded.firstStrip == fs);
        assert(decoded.range == rng);

        counter++;
      }
    }
    REQUIRE(true);
  }
}
