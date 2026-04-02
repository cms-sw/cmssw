/*
 *  makeprocess_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 5/18/05.
 *  Changed by Viji Sundararajan on 8-Jul-05.
 * 
 */

#include "catch2/catch_all.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakePyBind11ParameterSets.h"

#include <memory>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

TEST_CASE("ReadPsetsFrom", "[PythonParameterSet]") {
  SECTION("simpleTest") {
    const char* kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "dummy =  cms.PSet(b = cms.bool(True))\n"
        "foo = cms.PSet(a = cms.string('blah'))\n";
    std::shared_ptr<edm::ParameterSet> test = edm::cmspybind11::readPSetsFrom(kTest);

    REQUIRE(test->getParameterSet("dummy").getParameter<bool>("b") == true);
    REQUIRE(test->getParameterSet("foo").getParameter<std::string>("a") == std::string("blah"));
  }
}
