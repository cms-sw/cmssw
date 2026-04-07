/*----------------------------------------------------------------------

Test of the EventProcessor class.

----------------------------------------------------------------------*/
#include <exception>
#include <iostream>
#include <string>
#include <stdexcept>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "catch2/catch_all.hpp"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"

#include "oneapi/tbb/global_control.h"

// to be called also by the other catch2...
void doInit() {
  static bool firstTime = true;
  if (firstTime) {
    //std::cout << "common init" << std::endl;
    if (not edmplugin::PluginManager::isAvailable()) {
      edmplugin::PluginManager::configure(edmplugin::standard::config());
    }
    firstTime = false;
  }
}

void work() {
  //std::cout << "work in testeventprocessor2" << std::endl;
  std::string configuration(
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('PROD')\n"
      "process.maxEvents = cms.untracked.PSet(\n"
      "  input = cms.untracked.int32(5))\n"
      "process.source = cms.Source('EmptySource')\n"
      "process.m1 = cms.EDProducer('IntProducer',\n"
      "   ivalue = cms.int32(10))\n"
      "process.m2 = cms.EDProducer('ToyDoubleProducer',\n"
      "   dvalue = cms.double(3.3))\n"
      "process.out = cms.OutputModule('AsciiOutputModule')\n"
      "process.p1 = cms.Path(process.m1*process.m2)\n"
      "process.ep1 = cms.EndPath(process.out)");
  std::unique_ptr<edm::ParameterSet> pset = edm::getPSetFromConfig(configuration);
  edm::EventProcessor proc(std::move(pset));
  proc.beginJob();
  proc.run();
  proc.endJob();
}

TEST_CASE("EventProcessor2", "[Framework]") {
  static edm::propagate_const<std::unique_ptr<oneapi::tbb::global_control>> m_control;

  if (not m_control) {
    m_control = std::make_unique<oneapi::tbb::global_control>(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  }
  doInit();

  SECTION("eventprocessor2Test") {
    try {
      work();
    } catch (cms::Exception& e) {
      std::cerr << "CMS exception caught: " << e.explainSelf() << std::endl;
      REQUIRE("cms Exception caught in testeventprocessor2::eventprocessor2Test" == 0);
    } catch (std::runtime_error& e) {
      std::cerr << "Standard library exception caught: " << e.what() << std::endl;
      REQUIRE("std Exception caught in testeventprocessor2::eventprocessor2Test" == 0);
    } catch (...) {
      std::cerr << "Unknown exception caught" << std::endl;
      REQUIRE("unkown Exception caught in testeventprocessor2::eventprocessor2Test" == 0);
    }
  }
}
