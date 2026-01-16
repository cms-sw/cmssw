/*----------------------------------------------------------------------

Test program for edm::signalslot::Signal class.

 ----------------------------------------------------------------------*/

#include <catch2/catch_all.hpp>
#include <cassert>
#include <iostream>
#include <string>
#include "FWCore/Utilities/interface/Signal.h"

static int s_value = 0;
static void setValueFunct(int iValue) { s_value = iValue; }

TEST_CASE("edm::signalslot::Signal", "[Signal]") {
  SECTION("connectTest") {
    edm::signalslot::Signal<void(int)> sig;
    REQUIRE(sig.slots().size() == 0);

    int value1 = 0;
    sig.connect([&](int iValue) -> void { value1 = iValue; });
    REQUIRE(sig.slots().size() == 1);

    int value2 = 0;
    sig.connect([&](int iValue) { value2 = iValue; });
    REQUIRE(sig.slots().size() == 2);

    sig.connect(setValueFunct);
    //see that the slots we created are actually there
    for (auto const& slot : sig.slots()) {
      slot(5);
    }
    REQUIRE(value1 == 5);
    REQUIRE(value2 == 5);
    REQUIRE(value2 == s_value);
  }

  SECTION("emitTest") {
    edm::signalslot::Signal<void(int)> sig;

    int value1 = 0;
    sig.connect([&](int iValue) { value1 = iValue; });

    int value2 = 0;
    sig.connect([&](int iValue) { value2 = iValue; });

    sig.emit(5);
    REQUIRE(value1 == 5);
    REQUIRE(value2 == 5);

    sig.emit(1);
    REQUIRE(value1 == 1);
    REQUIRE(value2 == 1);
  }
}
