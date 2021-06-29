// -*- C++ -*-
//
// Package:     Core
// Class  :     unittest_fwmodelid
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jan 18 10:19:07 EST 2008
//

// system include files
#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <stdexcept>
#include <iostream>

// user include files
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWCompositeParameter.h"
#include "Fireworks/Core/interface/FWParameterizable.h"

namespace {
  struct Test : public FWParameterizable {
    Test()
        : m_double(this, "double", std::bind(&Test::doubleChanged, this, std::placeholders::_1)),
          m_long(this, "long", std::bind(&Test::longChanged, this, std::placeholders::_1)),
          m_wasChanged(false) {}

    void doubleChanged(double iValue) {
      BOOST_CHECK(iValue == m_double.value());
      m_wasChanged = true;
    }
    void longChanged(long iValue) {
      BOOST_CHECK(iValue == m_long.value());
      m_wasChanged = true;
    }

    FWDoubleParameter m_double;
    FWLongParameter m_long;
    bool m_wasChanged;
  };

  struct CompTest : public FWParameterizable {
    FWCompositeParameter m_comp;
    FWDoubleParameter m_d1;
    FWDoubleParameter m_d2;

    CompTest() : m_comp(this, "comp"), m_d1(&m_comp, "d1"), m_d2(&m_comp, "d2") {}
  };
}  // namespace
//
// constants, enums and typedefs
//
BOOST_AUTO_TEST_CASE(parameters) {
  Test t;
  BOOST_CHECK(0. == t.m_double.value());
  BOOST_CHECK(0 == t.m_long.value());
  BOOST_CHECK(not t.m_wasChanged);

  t.m_double.set(10.);
  BOOST_CHECK(10. == t.m_double.value());
  BOOST_CHECK(0 == t.m_long.value());
  BOOST_CHECK(t.m_wasChanged);

  t.m_wasChanged = false;
  t.m_long.set(2);
  BOOST_CHECK(10. == t.m_double.value());
  BOOST_CHECK(2 == t.m_long.value());
  BOOST_CHECK(t.m_wasChanged);

  //check configuration ability
  {
    FWConfiguration config;

    t.m_wasChanged = false;
    t.m_double.addTo(config);
    BOOST_CHECK(not t.m_wasChanged);

    BOOST_REQUIRE(0 != config.keyValues());
    BOOST_CHECK(1 == config.keyValues()->size());
    BOOST_CHECK(std::string("double") == config.keyValues()->front().first);

    t.m_double.set(1.);
    BOOST_CHECK(t.m_wasChanged);
    BOOST_CHECK(t.m_double.value() == 1.);

    t.m_wasChanged = false;
    t.m_double.setFrom(config);
    BOOST_CHECK(t.m_wasChanged);
    BOOST_CHECK(10. == t.m_double.value());
  }
  {
    FWConfiguration config;

    t.m_wasChanged = false;
    t.m_long.addTo(config);
    BOOST_CHECK(not t.m_wasChanged);

    BOOST_REQUIRE(0 != config.keyValues());
    BOOST_CHECK(1 == config.keyValues()->size());
    BOOST_CHECK(std::string("long") == config.keyValues()->front().first);

    t.m_long.set(1);
    BOOST_CHECK(t.m_wasChanged);
    BOOST_CHECK(t.m_long.value() == 1);

    t.m_wasChanged = false;
    t.m_long.setFrom(config);
    BOOST_CHECK(t.m_wasChanged);
    BOOST_CHECK(2 == t.m_long.value());
  }

  CompTest ct;
  BOOST_CHECK(ct.end() - ct.begin() == 1);
  BOOST_CHECK(ct.m_comp.end() - ct.m_comp.begin() == 2);
  ct.m_d1.set(10.);
  ct.m_d2.set(11.);

  FWConfiguration ctConf;
  ct.m_comp.addTo(ctConf);

  FWConfiguration::streamTo(std::cout, ctConf, "top");

  ct.m_d1.set(0);
  ct.m_d2.set(1);

  ct.m_comp.setFrom(ctConf);
  BOOST_CHECK(ct.m_d1.value() == 10.);
  BOOST_CHECK(ct.m_d2.value() == 11.);
}
