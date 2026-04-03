/*
 *  makeprocess_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 5/18/05.
 *
 */

#include "catch2/catch_all.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <memory>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <set>

static edm::ParameterSet modulePSet(std::string const& iLabel, std::string const& iType, std::string const& iCppType) {
  edm::ParameterSet temp;
  temp.addParameter("s", 1);
  temp.addParameter("@module_label", iLabel);
  temp.addParameter("@module_type", iType);
  temp.addParameter("@module_edm_type", iCppType);
  return temp;
}

typedef std::shared_ptr<edm::ParameterSet> ParameterSetPtr;
static ParameterSetPtr pSet(char const* c) {
  //ParameterSetPtr result( new edm::ProcessDesc(std::string(c)) );
  ParameterSetPtr result = PyBind11ProcessDesc(std::string(c), false).parameterSet();
  REQUIRE(result->getParameter<std::string>("@process_name") == "test");
  return result;
}

TEST_CASE("MakeProcess", "[PythonParameterSet]") {
  SECTION("simpleProcess") {
    char const* kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('test')\n"
        "dummy =  cms.PSet(b = cms.bool(True))\n";
    ParameterSetPtr test = pSet(kTest);
  }

  SECTION("using") {
    char const* kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('test')\n"
        "process.dummy = cms.PSet(\n"
        "    b = cms.bool(True)\n"
        ")\n"
        "process.dummy2 = cms.PSet(\n"
        "    d = cms.bool(True)\n"
        ")\n"
        "process.m1 = cms.EDFilter('Dummy',\n"
        "    process.dummy\n"
        ")\n"
        "process.m2 = cms.EDFilter('Dummy2',\n"
        "    process.dummy2\n"
        ")\n"
        "process.p = cms.Path(process.m1+process.m2)\n";

    ParameterSetPtr test = pSet(kTest);

    //REQUIRE(test->getParameterSet("dummy").getBool("b") == true);
    REQUIRE(test->getParameterSet("m1").getParameter<bool>("b") == true);
    REQUIRE(test->getParameterSet("m2").getParameter<bool>("d") == true);
  }

  SECTION("path") {
    char const* kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('test')\n"
        "process.cone5 = cms.EDFilter('PhonyConeJet',\n"
        "    i = cms.int32(5)\n"
        ")\n"
        "process.cone7 = cms.EDFilter('PhonyConeJet',\n"
        "    i = cms.int32(7)\n"
        ")\n"
        "process.jtanalyzer = cms.EDAnalyzer('jtanalyzer')\n"
        "process.writer = cms.OutputModule('Writer')\n"
        "process.cones = cms.Sequence(process.cone5*process.cone7)\n"
        "process.term1 = cms.Path(process.cones+process.jtanalyzer)\n"
        "process.atEnd = cms.EndPath(process.writer)\n";

    ParameterSetPtr test = pSet(kTest);
    //REQUIRE(test->pathFragments().size() == 5);

    edm::ParameterSet& myparams = *(test);
    myparams.registerIt();
    std::string rep = myparams.toString();
    edm::ParameterSet copy(rep);
    REQUIRE(copy == myparams);
  }

  SECTION("module") {
    char const* kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('test')\n"
        "process.cones = cms.EDFilter('Module',\n"
        "    s = cms.int32(1)\n"
        ")\n"
        "process.NoLabelModule = cms.ESProducer('NoLabelModule',\n"
        "    s = cms.int32(1)\n"
        ")\n"
        "process.labeled = cms.ESProducer('LabelModule',\n"
        "    s = cms.int32(1)\n"
        ")\n"
        "process.source = cms.Source('InputSource',\n"
        "    s = cms.int32(1)\n"
        ")\n"
        "process.NoLabelRetriever = cms.ESSource('NoLabelRetriever',\n"
        "    s = cms.int32(1)\n"
        ")\n"
        "process.label = cms.ESSource('LabelRetriever',\n"
        "    s = cms.int32(1)\n"
        ")\n"
        "process.p = cms.Path(process.cones)\n";

    ParameterSetPtr test = pSet(kTest);

    static edm::ParameterSet const kEmpty;
    edm::ParameterSet const kCone(modulePSet("cones", "Module", "EDFilter"));
    std::ostringstream out;
    out << kCone.toString() << std::endl;
    out << test->getParameterSet("cones").toString() << std::endl;

    edm::ParameterSet const kMainInput(modulePSet("@main_input", "InputSource", "Source"));

    edm::ParameterSet const kNoLabelModule(modulePSet("", "NoLabelModule", "ESProducer"));
    edm::ParameterSet const kLabelModule(modulePSet("labeled", "LabelModule", "ESProducer"));
    edm::ParameterSet const kNoLabelRetriever(modulePSet("", "NoLabelRetriever", "ESSource"));
    edm::ParameterSet const kLabelRetriever(modulePSet("label", "LabelRetriever", "ESSource"));

    REQUIRE(kEmpty != (test->getParameterSet("cones")));
    REQUIRE(kCone == test->getParameterSet("cones"));

    REQUIRE(kEmpty != (test->getParameterSet("@main_input")));
    REQUIRE(kMainInput == (test->getParameterSet("@main_input")));

    REQUIRE(kEmpty != (test->getParameterSet("NoLabelModule@")));
    REQUIRE(kNoLabelModule == test->getParameterSet("NoLabelModule@"));

    REQUIRE(kEmpty != (test->getParameterSet("LabelModule@labeled")));
    REQUIRE(kLabelModule == test->getParameterSet("LabelModule@labeled"));

    REQUIRE(kEmpty != (test->getParameterSet("NoLabelRetriever@")));
    REQUIRE(kNoLabelRetriever == test->getParameterSet("NoLabelRetriever@"));

    REQUIRE(kEmpty != (test->getParameterSet("LabelRetriever@label")));
    REQUIRE(kLabelRetriever == test->getParameterSet("LabelRetriever@label"));
  }

  SECTION("emptyModule") {
    char const* kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('test')\n"
        "process.thing = cms.EDFilter('XX')\n"
        "process.p = cms.Path(process.thing)\n";

    ParameterSetPtr test = pSet(kTest);

    edm::ParameterSet& myparams = *(test);
    myparams.registerIt();
    std::string rep = myparams.toString();
    edm::ParameterSet copy(rep);
    REQUIRE(copy == myparams);
  }

  SECTION("task") {
    char const* kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('test')\n"
        "process.load(\"FWCore.PythonParameterSet.test.testTask_cff\")\n"
        "t10 = cms.Task(process.m29, process.m30, process.f29, process.f30,"
        "process.ess27, process.ess28, process.esp27, process.esp28,"
        "process.serv27, process.serv28)\n";

    ParameterSetPtr test = pSet(kTest);

    REQUIRE(!test->existsAs<edm::ParameterSet>("m1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m2"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m3"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m4"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m5"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m6"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m7"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m8"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m9"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m10"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m11"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m12"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m13"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m14"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m15"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m16"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m17"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m18"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m19"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m20"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m21"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m22"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m23"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m24"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m25"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m26"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m27"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m28"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m29"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m30"));

    REQUIRE(!test->existsAs<edm::ParameterSet>("f1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f2"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f3"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f4"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f5"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f6"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f7"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f8"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f9"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f10"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f11"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f12"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f13"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f14"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f15"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f16"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f17"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f18"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f19"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f20"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f21"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f22"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f23"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f24"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f25"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f26"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f27"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f28"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f29"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f30"));

    REQUIRE(!test->existsAs<edm::ParameterSet>("a1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a2"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a3"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a4"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a5"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a6"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a7"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a8"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a9"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a10"));

    REQUIRE(!test->existsAs<edm::ParameterSet>("o1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("o2"));
    REQUIRE(test->existsAs<edm::ParameterSet>("o7"));
    REQUIRE(test->existsAs<edm::ParameterSet>("o8"));
    REQUIRE(test->existsAs<edm::ParameterSet>("o9"));
    REQUIRE(test->existsAs<edm::ParameterSet>("o10"));

    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess1"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess2@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess@ess3"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess4@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess11"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess12@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess13"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess14@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess15"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess16@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess17"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess18@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess19"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess20@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess21"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess22@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess23"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess24@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess25"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess26@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess27"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess28@"));

    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp1"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp2@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp@esp3"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp4@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp11"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp12@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp13"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp14@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp15"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp16@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp17"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp18@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp19"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp20@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp21"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp22@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp23"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp24@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp25"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp26@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp27"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp28@"));

    std::vector<edm::ParameterSet> const& vpsetServices = test->getUntrackedParameterSetVector("services");
    // Note that the vector<ParameterSet> is not sorted. The order
    // depends on the order of a python iteration through a dictionary
    // which could be anything.
    std::set<std::string> serviceNames;
    for (auto const& pset : vpsetServices) {
      serviceNames.insert(pset.getParameter<std::string>("@service_type"));
    }
    std::vector<std::string> expectedServiceNames{"MessageLogger", "serv1",  "serv2",  "serv11", "serv12", "serv13",
                                                  "serv14",        "serv15", "serv16", "serv17", "serv18", "serv19",
                                                  "serv20",        "serv21", "serv22", "serv23", "serv24", "serv25",
                                                  "serv26",        "serv27", "serv28"};
    bool result = true;
    for (auto const& name : expectedServiceNames) {
      if (serviceNames.find(name) == serviceNames.end()) {
        result = false;
      }
    }
    if (serviceNames.size() != expectedServiceNames.size()) {
      result = false;
    }
    REQUIRE(result);
  }

  SECTION("taskWithEmptySchedule") {
    char const* kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('test')\n"
        "process.load(\"FWCore.PythonParameterSet.test.testTask_cff\")\n"
        "t10 = cms.Task(process.m29, process.m30, process.f29, process.f30,"
        "process.ess27, process.ess28, process.esp27, process.esp28,"
        "process.serv27, process.serv28)\n"
        "process.schedule = cms.Schedule()\n";

    ParameterSetPtr test = pSet(kTest);

    REQUIRE(!test->existsAs<edm::ParameterSet>("m1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m2"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m3"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m4"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m5"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m6"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m7"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m8"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m9"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m10"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m11"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m12"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m13"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m14"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m15"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m16"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m17"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m18"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m19"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m20"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m21"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m22"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m23"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m24"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m25"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m26"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m27"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m28"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m29"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m30"));

    REQUIRE(!test->existsAs<edm::ParameterSet>("f1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f2"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f3"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f4"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f5"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f6"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f7"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f8"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f9"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f10"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f11"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f12"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f13"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f14"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f15"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f16"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f17"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f18"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f19"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f20"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f21"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f22"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f23"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f24"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f25"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f26"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f27"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f28"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f29"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f30"));

    REQUIRE(!test->existsAs<edm::ParameterSet>("a1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a2"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a3"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a4"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a5"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a6"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a7"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a8"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a9"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a10"));

    REQUIRE(!test->existsAs<edm::ParameterSet>("o1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("o2"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("o7"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("o8"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("o9"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("o10"));

    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess1"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess2@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess@ess3"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess4@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess@ess11"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess12@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess@ess13"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess14@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess@ess15"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess16@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess@ess17"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess18@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess@ess19"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess20@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess@ess21"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess22@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess@ess23"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess24@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess@ess25"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("ess26@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess27"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess28@"));

    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp1"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp2@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp@esp3"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp4@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp@esp11"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp12@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp@esp13"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp14@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp@esp15"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp16@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp@esp17"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp18@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp@esp19"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp20@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp@esp21"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp22@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp@esp23"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp24@"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp@esp25"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("esp26@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp27"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp28@"));

    std::vector<edm::ParameterSet> const& vpsetServices = test->getUntrackedParameterSetVector("services");
    // Note that the vector<ParameterSet> is not sorted. The order
    // depends on the order of a python iteration through a dictionary
    // which could be anything.
    std::set<std::string> serviceNames;
    for (auto const& pset : vpsetServices) {
      serviceNames.insert(pset.getParameter<std::string>("@service_type"));
    }
    std::vector<std::string> expectedServiceNames{"MessageLogger", "serv1", "serv2", "serv27", "serv28"};
    bool result = true;
    for (auto const& name : expectedServiceNames) {
      if (serviceNames.find(name) == serviceNames.end()) {
        result = false;
      }
    }
    if (serviceNames.size() != expectedServiceNames.size()) {
      result = false;
    }
    REQUIRE(result);
  }

  SECTION("taskWithSchedule") {
    char const* kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('test')\n"
        "process.load(\"FWCore.PythonParameterSet.test.testTask_cff\")\n"
        "t10 = cms.Task(process.m29, process.m30, process.f29, process.f30,"
        "process.ess27, process.ess28, process.esp27, process.esp28,"
        "process.serv27, process.serv28)\n"
        "process.schedule = cms.Schedule(process.p1, process.p2, process.e1, process.e2,"
        "process.pf1, process.pf2, process.ef1, process.ef2,"
        "process.pa1, process.pa2, process.ea1, process.ea2,"
        "process.eo1, process.eo2, process.pess2, process.eess2,"
        "process.pesp2, process.eesp2, process.pserv2, process.eserv2,"
        "tasks=[process.t9, process.tf9, process.tess10,process.tesp10,"
        "process.tserv10])\n";

    ParameterSetPtr test = pSet(kTest);

    REQUIRE(!test->existsAs<edm::ParameterSet>("m1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m2"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m3"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m4"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m5"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m6"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m7"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m8"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m9"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m10"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m11"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m12"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m13"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m14"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m15"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m16"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m17"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m18"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m19"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m20"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m21"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m22"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m23"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m24"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m25"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m26"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m27"));
    REQUIRE(test->existsAs<edm::ParameterSet>("m28"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m29"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("m30"));

    REQUIRE(!test->existsAs<edm::ParameterSet>("f1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f2"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f3"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f4"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f5"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f6"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f7"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f8"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f9"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f10"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f11"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f12"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f13"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f14"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f15"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f16"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f17"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f18"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f19"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f20"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f21"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f22"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f23"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f24"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f25"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f26"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f27"));
    REQUIRE(test->existsAs<edm::ParameterSet>("f28"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f29"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("f30"));

    REQUIRE(!test->existsAs<edm::ParameterSet>("a1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("a2"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a3"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a4"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a5"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a6"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a7"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a8"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a9"));
    REQUIRE(test->existsAs<edm::ParameterSet>("a10"));

    REQUIRE(!test->existsAs<edm::ParameterSet>("o1"));
    REQUIRE(!test->existsAs<edm::ParameterSet>("o2"));
    REQUIRE(test->existsAs<edm::ParameterSet>("o7"));
    REQUIRE(test->existsAs<edm::ParameterSet>("o8"));
    REQUIRE(test->existsAs<edm::ParameterSet>("o9"));
    REQUIRE(test->existsAs<edm::ParameterSet>("o10"));

    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess1"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess2@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess3"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess4@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess11"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess12@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess13"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess14@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess15"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess16@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess17"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess18@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess19"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess20@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess21"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess22@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess23"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess24@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess25"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess26@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess@ess27"));
    REQUIRE(test->existsAs<edm::ParameterSet>("ess28@"));

    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp1"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp2@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp3"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp4@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp11"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp12@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp13"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp14@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp15"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp16@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp17"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp18@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp19"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp20@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp21"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp22@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp23"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp24@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp25"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp26@"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp@esp27"));
    REQUIRE(test->existsAs<edm::ParameterSet>("esp28@"));

    std::vector<edm::ParameterSet> const& vpsetServices = test->getUntrackedParameterSetVector("services");
    // Note that the vector<ParameterSet> is not sorted. The order
    // depends on the order of a python iteration through a dictionary
    // which could be anything.
    std::set<std::string> serviceNames;
    for (auto const& pset : vpsetServices) {
      serviceNames.insert(pset.getParameter<std::string>("@service_type"));
    }
    std::vector<std::string> expectedServiceNames{"MessageLogger", "serv1",  "serv2",  "serv3",  "serv4",  "serv11",
                                                  "serv12",        "serv13", "serv14", "serv15", "serv16", "serv17",
                                                  "serv18",        "serv19", "serv20", "serv21", "serv22", "serv23",
                                                  "serv24",        "serv25", "serv26", "serv27", "serv28"};
    bool result = true;
    for (auto const& name : expectedServiceNames) {
      if (serviceNames.find(name) == serviceNames.end()) {
        result = false;
      }
    }
    if (serviceNames.size() != expectedServiceNames.size()) {
      result = false;
    }
    REQUIRE(result);
  }

  SECTION("edmException") {
    {
      char const* const kTest =
          "import FWCore.ParameterSet.Config as cms\n"
          "raise cms.EDMException(cms.edm.errors.Configuration,'test message')\n";

      bool exceptionHappened = false;
      try {
        (void)pSet(kTest);
      } catch (edm::Exception const& e) {
        exceptionHappened = true;
        REQUIRE(e.categoryCode() == edm::errors::Configuration);
      } catch (std::exception const& e) {
        std::cout << "wrong error " << e.what() << std::endl;
        exceptionHappened = true;
        REQUIRE(false);
      }
      REQUIRE(exceptionHappened);
    }

    {
      char const* const kTest =
          "import FWCore.ParameterSet.Config as cms\n"
          "raise cms.EDMException(cms.edm.errors.UnavailableAccelerator,'test message')\n";

      bool exceptionHappened = false;
      try {
        (void)pSet(kTest);
      } catch (edm::Exception const& e) {
        exceptionHappened = true;
        REQUIRE(e.categoryCode() == edm::errors::UnavailableAccelerator);
      } catch (std::exception const& e) {
        std::cout << "wrong error " << e.what() << std::endl;
        exceptionHappened = true;
        REQUIRE(false);
      }
      REQUIRE(exceptionHappened);
    }
  }
}
/*
void testmakeprocess::windowsLineEndingTest() {

  std::ostringstream oss;
  char const ret = '\r';
  char const nl = '\n';
  char const dquote = '"';
  char const backsl = '\\';

  oss << ret << nl
      << "import FWCore.ParameterSet.Config as cms" << ret << nl
      << "process = cms.Process('test')" << ret << nl
      << "  source = cms.Source('InputSource'," << ret << nl
      << "    i=cms.int32(1)" << ret << nl
      << "    s1 = cms.string(" << dquote << ret << dquote <<  ')' <<ret << nl
      << "    s2 = cms.string(" << dquote << backsl << backsl << 'r' << dquote << ')' << ret << nl
      << "  )" << ret << nl;
  char const* kTest = oss.str().c_str();
  std::cerr << "\n------------------------------\n";
  std::cerr << "s1 will look funky because of the embedded return\n";
  std::cerr << "s2 shows how to get the chars backslash-r into a string\n";
  std::cerr << kTest;
  std::cerr << "\n------------------------------\n";

   ParameterSetPtr test = pSet(kTest);

   edm::ParameterSet const& p = *(test->getProcessPSet());

   edm::ParameterSet src = p.getParameterSet("@main_input");
   REQUIRE(src.getParameter<int>("i") == 1);
   std::string s1 = src.getParameter<std::string>("s1");
   std::string s2 = src.getParameter<std::string>("s2");

   std::cerr << "\nsize of s1 is: " << s1.size();
   std::cerr << "\nsize of s2 is: " << s2.size() << '\n';

   REQUIRE(s1.size() == 1);
   REQUIRE(s1[0] == ret);

   REQUIRE(s2.size() == 2);
   REQUIRE(s2[0] == backsl);
   REQUIRE(s2[1] == 'r');
}
*/
