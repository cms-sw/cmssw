/*----------------------------------------------------------------------

Test program for edm::signalslot::Signal class.

 ----------------------------------------------------------------------*/

#include <catch2/catch_all.hpp>
#include <cassert>
#include <iostream>
#include <string>
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Utilities/interface/SignalSentry.h"

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

TEST_CASE("edm::signalslot::SignalSentry", "[Signal]") {
  SECTION("sentryTest") {
    edm::signalslot::Signal<void(int)> sig;

    int value1 = 0;
    sig.connect([&](int iValue) { value1 = iValue; });

    int value2 = 0;
    sig.connect([&](int iValue) { value2 = iValue; });

    {
      auto sentry = edm::signalslot::make_sentry([&]() { sig.emit(5); });
      sentry.succeeded();
    }
    REQUIRE(value1 == 5);
    REQUIRE(value2 == 5);

    {
      auto sentry = edm::signalslot::make_sentry([&]() { sig.emit(1); });
      // Do not call succeeded(), so the signal will be emitted in the destructor.
    }
    REQUIRE(value1 == 1);
    REQUIRE(value2 == 1);
  }

  SECTION("sentryCallsSucceededManyTimesTest") {
    edm::signalslot::Signal<void(int)> sig;

    int value1 = 0;
    sig.connect([&](int iValue) { value1 = iValue; });

    int value2 = 0;
    sig.connect([&](int iValue) { value2 = iValue; });

    {
      auto sentry = edm::signalslot::make_sentry([&]() { sig.emit(5); });
      sentry.succeeded();
      sentry.succeeded();  // not intended to be called more than once, but should behave well nevertheless
    }
    REQUIRE(value1 == 5);
    REQUIRE(value2 == 5);
  }

  SECTION("sentryExceptionTest") {
    edm::signalslot::Signal<void(int)> sig;

    int value1 = 0;
    sig.connect([&](int iValue) { value1 = iValue; });

    int value2 = 0;
    sig.connect([&](int iValue) { value2 = iValue; });

    try {
      auto sentry = edm::signalslot::make_sentry([&]() { sig.emit(5); });
      throw std::runtime_error("test exception");
      sentry.succeeded();
    } catch (std::exception const&) {
      // Ignore exception
    }
    REQUIRE(value1 == 5);
    REQUIRE(value2 == 5);
  }

  SECTION("sentryExceptionInSignalTest") {
    edm::signalslot::Signal<void(int)> sig;

    int value1 = 0;
    sig.connect([&](int iValue) { value1 = iValue; });

    sig.connect([&](int iValue) { throw std::runtime_error("test exception in signal"); });

    int value2 = 0;
    sig.connect([&](int iValue) { value2 = iValue; });

    SECTION("Operation succeeds") {
      bool exceptionCaught = false;
      try {
        auto sentry = edm::signalslot::make_sentry([&]() { sig.emit(5); });
        sentry.succeeded();
      } catch (std::exception const&) {
        // The exception from the signal should be propagated.
        exceptionCaught = true;
      }
      REQUIRE(value1 == 5);
      REQUIRE(value2 == 5);  // The second action is called even if a preceding action throws an exception.
      REQUIRE(exceptionCaught == true);
    }

    SECTION("Succeeded forgotten") {
      bool exceptionCaught = false;
      try {
        auto sentry = edm::signalslot::make_sentry([&]() { sig.emit(5); });
      } catch (std::exception const&) {
        exceptionCaught = true;
      }
      REQUIRE(value1 == 5);
      REQUIRE(value2 == 5);
      // Sentry destructor should not throw an exception
      REQUIRE(exceptionCaught == false);
    }

    SECTION("Operation throws an exception") {
      bool exceptionCaught = false;
      try {
        auto sentry = edm::signalslot::make_sentry([&]() { sig.emit(5); });
        throw std::runtime_error("test exception");
        sentry.succeeded();
      } catch (std::exception const&) {
        // Ignore the exception from the operation.
        exceptionCaught = true;
      }
      REQUIRE(value1 == 5);
      REQUIRE(value2 == 5);
      REQUIRE(exceptionCaught == true);
    }
  }

  SECTION("sentryIfTest") {
    edm::signalslot::Signal<void(int)> sig;

    int value1 = 0;
    sig.connect([&](int iValue) { value1 = iValue; });

    int value2 = 0;
    sig.connect([&](int iValue) { value2 = iValue; });

    {
      auto sentry = edm::signalslot::make_sentry_if(true, [&]() { sig.emit(5); });
      sentry.succeeded();
    }
    REQUIRE(value1 == 5);
    REQUIRE(value2 == 5);

    {
      auto sentry = edm::signalslot::make_sentry_if(false, [&]() { sig.emit(1); });
      sentry.succeeded();
    }
    REQUIRE(value1 == 5);
    REQUIRE(value2 == 5);
  }
}
