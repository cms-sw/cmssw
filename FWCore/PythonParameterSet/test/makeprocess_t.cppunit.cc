/*
 *  makeprocess_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 5/18/05.
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cppunit/extensions/HelperMacros.h>

#include <memory>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <set>

class testmakeprocess : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testmakeprocess);
  CPPUNIT_TEST(simpleProcessTest);
  CPPUNIT_TEST(usingTest);
  CPPUNIT_TEST(pathTest);
  CPPUNIT_TEST(moduleTest);
  CPPUNIT_TEST(emptyModuleTest);
  CPPUNIT_TEST(taskTest);
  CPPUNIT_TEST(taskTestWithEmptySchedule);
  CPPUNIT_TEST(taskTestWithSchedule);
  CPPUNIT_TEST(edmException);
  //CPPUNIT_TEST(windowsLineEndingTest);
  //CPPUNIT_TEST_EXCEPTION(emptyPsetTest,edm::Exception);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void simpleProcessTest();
  void usingTest();
  void pathTest();
  void moduleTest();
  void emptyModuleTest();
  void taskTest();
  void taskTestWithEmptySchedule();
  void taskTestWithSchedule();
  void edmException();
  //void windowsLineEndingTest();
private:
  typedef std::shared_ptr<edm::ParameterSet> ParameterSetPtr;
  ParameterSetPtr pSet(char const* c) {
    //ParameterSetPtr result( new edm::ProcessDesc(std::string(c)) );
    ParameterSetPtr result = PyBind11ProcessDesc(std::string(c)).parameterSet();
    CPPUNIT_ASSERT(result->getParameter<std::string>("@process_name") == "test");
    return result;
  }
  //  void emptyPsetTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testmakeprocess);

void testmakeprocess::simpleProcessTest() {
  char const* kTest =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('test')\n"
      "dummy =  cms.PSet(b = cms.bool(True))\n";
  ParameterSetPtr test = pSet(kTest);
}

void testmakeprocess::usingTest() {
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

  //CPPUNIT_ASSERT(test->getParameterSet("dummy").getBool("b") == true);
  CPPUNIT_ASSERT(test->getParameterSet("m1").getParameter<bool>("b") == true);
  CPPUNIT_ASSERT(test->getParameterSet("m2").getParameter<bool>("d") == true);
}

void testmakeprocess::pathTest() {
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
  //CPPUNIT_ASSERT(test->pathFragments().size() == 5);

  edm::ParameterSet& myparams = *(test);
  myparams.registerIt();
  std::string rep = myparams.toString();
  edm::ParameterSet copy(rep);
  CPPUNIT_ASSERT(copy == myparams);
}

edm::ParameterSet modulePSet(std::string const& iLabel, std::string const& iType, std::string const& iCppType) {
  edm::ParameterSet temp;
  temp.addParameter("s", 1);
  temp.addParameter("@module_label", iLabel);
  temp.addParameter("@module_type", iType);
  temp.addParameter("@module_edm_type", iCppType);
  return temp;
}

void testmakeprocess::moduleTest() {
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

  CPPUNIT_ASSERT(kEmpty != (test->getParameterSet("cones")));
  CPPUNIT_ASSERT(kCone == test->getParameterSet("cones"));

  CPPUNIT_ASSERT(kEmpty != (test->getParameterSet("@main_input")));
  CPPUNIT_ASSERT(kMainInput == (test->getParameterSet("@main_input")));

  CPPUNIT_ASSERT(kEmpty != (test->getParameterSet("NoLabelModule@")));
  CPPUNIT_ASSERT(kNoLabelModule == test->getParameterSet("NoLabelModule@"));

  CPPUNIT_ASSERT(kEmpty != (test->getParameterSet("LabelModule@labeled")));
  CPPUNIT_ASSERT(kLabelModule == test->getParameterSet("LabelModule@labeled"));

  CPPUNIT_ASSERT(kEmpty != (test->getParameterSet("NoLabelRetriever@")));
  CPPUNIT_ASSERT(kNoLabelRetriever == test->getParameterSet("NoLabelRetriever@"));

  CPPUNIT_ASSERT(kEmpty != (test->getParameterSet("LabelRetriever@label")));
  CPPUNIT_ASSERT(kLabelRetriever == test->getParameterSet("LabelRetriever@label"));
}

void testmakeprocess::emptyModuleTest() {
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
  CPPUNIT_ASSERT(copy == myparams);
}

void testmakeprocess::taskTest() {
  char const* kTest =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('test')\n"
      "process.load(\"FWCore.PythonParameterSet.test.testTask_cff\")\n"
      "t10 = cms.Task(process.m29, process.m30, process.f29, process.f30,"
      "process.ess27, process.ess28, process.esp27, process.esp28,"
      "process.serv27, process.serv28)\n";

  ParameterSetPtr test = pSet(kTest);

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("m1") && !test->existsAs<edm::ParameterSet>("m2") &&
                 test->existsAs<edm::ParameterSet>("m3") && test->existsAs<edm::ParameterSet>("m4") &&
                 test->existsAs<edm::ParameterSet>("m5") && test->existsAs<edm::ParameterSet>("m6") &&
                 test->existsAs<edm::ParameterSet>("m7") && test->existsAs<edm::ParameterSet>("m8") &&
                 test->existsAs<edm::ParameterSet>("m9") && test->existsAs<edm::ParameterSet>("m10") &&
                 test->existsAs<edm::ParameterSet>("m11") && test->existsAs<edm::ParameterSet>("m12") &&
                 test->existsAs<edm::ParameterSet>("m13") && test->existsAs<edm::ParameterSet>("m14") &&
                 test->existsAs<edm::ParameterSet>("m15") && test->existsAs<edm::ParameterSet>("m16") &&
                 test->existsAs<edm::ParameterSet>("m17") && test->existsAs<edm::ParameterSet>("m18") &&
                 test->existsAs<edm::ParameterSet>("m19") && test->existsAs<edm::ParameterSet>("m20") &&
                 test->existsAs<edm::ParameterSet>("m21") && test->existsAs<edm::ParameterSet>("m22") &&
                 test->existsAs<edm::ParameterSet>("m23") && test->existsAs<edm::ParameterSet>("m24") &&
                 test->existsAs<edm::ParameterSet>("m25") && test->existsAs<edm::ParameterSet>("m26") &&
                 !test->existsAs<edm::ParameterSet>("m27") && !test->existsAs<edm::ParameterSet>("m28") &&
                 !test->existsAs<edm::ParameterSet>("m29") && !test->existsAs<edm::ParameterSet>("m30"));

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("f1") && !test->existsAs<edm::ParameterSet>("f2") &&
                 test->existsAs<edm::ParameterSet>("f3") && test->existsAs<edm::ParameterSet>("f4") &&
                 test->existsAs<edm::ParameterSet>("f5") && test->existsAs<edm::ParameterSet>("f6") &&
                 test->existsAs<edm::ParameterSet>("f7") && test->existsAs<edm::ParameterSet>("f8") &&
                 test->existsAs<edm::ParameterSet>("f9") && test->existsAs<edm::ParameterSet>("f10") &&
                 test->existsAs<edm::ParameterSet>("f11") && test->existsAs<edm::ParameterSet>("f12") &&
                 test->existsAs<edm::ParameterSet>("f13") && test->existsAs<edm::ParameterSet>("f14") &&
                 test->existsAs<edm::ParameterSet>("f15") && test->existsAs<edm::ParameterSet>("f16") &&
                 test->existsAs<edm::ParameterSet>("f17") && test->existsAs<edm::ParameterSet>("f18") &&
                 test->existsAs<edm::ParameterSet>("f19") && test->existsAs<edm::ParameterSet>("f20") &&
                 test->existsAs<edm::ParameterSet>("f21") && test->existsAs<edm::ParameterSet>("f22") &&
                 test->existsAs<edm::ParameterSet>("f23") && test->existsAs<edm::ParameterSet>("f24") &&
                 test->existsAs<edm::ParameterSet>("f25") && test->existsAs<edm::ParameterSet>("f26") &&
                 !test->existsAs<edm::ParameterSet>("f27") && !test->existsAs<edm::ParameterSet>("f28") &&
                 !test->existsAs<edm::ParameterSet>("f29") && !test->existsAs<edm::ParameterSet>("f30"));

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("a1") && !test->existsAs<edm::ParameterSet>("a2") &&
                 test->existsAs<edm::ParameterSet>("a3") && test->existsAs<edm::ParameterSet>("a4") &&
                 test->existsAs<edm::ParameterSet>("a5") && test->existsAs<edm::ParameterSet>("a6") &&
                 test->existsAs<edm::ParameterSet>("a7") && test->existsAs<edm::ParameterSet>("a8") &&
                 test->existsAs<edm::ParameterSet>("a9") && test->existsAs<edm::ParameterSet>("a10"));

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("o1") && !test->existsAs<edm::ParameterSet>("o2") &&
                 test->existsAs<edm::ParameterSet>("o7") && test->existsAs<edm::ParameterSet>("o8") &&
                 test->existsAs<edm::ParameterSet>("o9") && test->existsAs<edm::ParameterSet>("o10"));

  CPPUNIT_ASSERT(test->existsAs<edm::ParameterSet>("ess@ess1") && test->existsAs<edm::ParameterSet>("ess2@") &&
                 !test->existsAs<edm::ParameterSet>("ess@ess3") && !test->existsAs<edm::ParameterSet>("ess4@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess11") && test->existsAs<edm::ParameterSet>("ess12@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess13") && test->existsAs<edm::ParameterSet>("ess14@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess15") && test->existsAs<edm::ParameterSet>("ess16@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess17") && test->existsAs<edm::ParameterSet>("ess18@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess19") && test->existsAs<edm::ParameterSet>("ess20@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess21") && test->existsAs<edm::ParameterSet>("ess22@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess23") && test->existsAs<edm::ParameterSet>("ess24@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess25") && test->existsAs<edm::ParameterSet>("ess26@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess27") && test->existsAs<edm::ParameterSet>("ess28@"));

  CPPUNIT_ASSERT(test->existsAs<edm::ParameterSet>("esp@esp1") && test->existsAs<edm::ParameterSet>("esp2@") &&
                 !test->existsAs<edm::ParameterSet>("esp@esp3") && !test->existsAs<edm::ParameterSet>("esp4@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp11") && test->existsAs<edm::ParameterSet>("esp12@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp13") && test->existsAs<edm::ParameterSet>("esp14@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp15") && test->existsAs<edm::ParameterSet>("esp16@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp17") && test->existsAs<edm::ParameterSet>("esp18@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp19") && test->existsAs<edm::ParameterSet>("esp20@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp21") && test->existsAs<edm::ParameterSet>("esp22@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp23") && test->existsAs<edm::ParameterSet>("esp24@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp25") && test->existsAs<edm::ParameterSet>("esp26@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp27") && test->existsAs<edm::ParameterSet>("esp28@"));

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
  CPPUNIT_ASSERT(result);
}

void testmakeprocess::taskTestWithEmptySchedule() {
  char const* kTest =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('test')\n"
      "process.load(\"FWCore.PythonParameterSet.test.testTask_cff\")\n"
      "t10 = cms.Task(process.m29, process.m30, process.f29, process.f30,"
      "process.ess27, process.ess28, process.esp27, process.esp28,"
      "process.serv27, process.serv28)\n"
      "process.schedule = cms.Schedule()\n";

  ParameterSetPtr test = pSet(kTest);

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("m1") && !test->existsAs<edm::ParameterSet>("m2") &&
                 !test->existsAs<edm::ParameterSet>("m3") && !test->existsAs<edm::ParameterSet>("m4") &&
                 !test->existsAs<edm::ParameterSet>("m5") && !test->existsAs<edm::ParameterSet>("m6") &&
                 !test->existsAs<edm::ParameterSet>("m7") && !test->existsAs<edm::ParameterSet>("m8") &&
                 !test->existsAs<edm::ParameterSet>("m9") && !test->existsAs<edm::ParameterSet>("m10") &&
                 !test->existsAs<edm::ParameterSet>("m11") && !test->existsAs<edm::ParameterSet>("m12") &&
                 !test->existsAs<edm::ParameterSet>("m13") && !test->existsAs<edm::ParameterSet>("m14") &&
                 !test->existsAs<edm::ParameterSet>("m15") && !test->existsAs<edm::ParameterSet>("m16") &&
                 !test->existsAs<edm::ParameterSet>("m17") && !test->existsAs<edm::ParameterSet>("m18") &&
                 !test->existsAs<edm::ParameterSet>("m19") && !test->existsAs<edm::ParameterSet>("m20") &&
                 !test->existsAs<edm::ParameterSet>("m21") && !test->existsAs<edm::ParameterSet>("m22") &&
                 !test->existsAs<edm::ParameterSet>("m23") && !test->existsAs<edm::ParameterSet>("m24") &&
                 !test->existsAs<edm::ParameterSet>("m25") && !test->existsAs<edm::ParameterSet>("m26") &&
                 !test->existsAs<edm::ParameterSet>("m27") && !test->existsAs<edm::ParameterSet>("m28") &&
                 !test->existsAs<edm::ParameterSet>("m29") && !test->existsAs<edm::ParameterSet>("m30"));

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("f1") && !test->existsAs<edm::ParameterSet>("f2") &&
                 !test->existsAs<edm::ParameterSet>("f3") && !test->existsAs<edm::ParameterSet>("f4") &&
                 !test->existsAs<edm::ParameterSet>("f5") && !test->existsAs<edm::ParameterSet>("f6") &&
                 !test->existsAs<edm::ParameterSet>("f7") && !test->existsAs<edm::ParameterSet>("f8") &&
                 !test->existsAs<edm::ParameterSet>("f9") && !test->existsAs<edm::ParameterSet>("f10") &&
                 !test->existsAs<edm::ParameterSet>("f11") && !test->existsAs<edm::ParameterSet>("f12") &&
                 !test->existsAs<edm::ParameterSet>("f13") && !test->existsAs<edm::ParameterSet>("f14") &&
                 !test->existsAs<edm::ParameterSet>("f15") && !test->existsAs<edm::ParameterSet>("f16") &&
                 !test->existsAs<edm::ParameterSet>("f17") && !test->existsAs<edm::ParameterSet>("f18") &&
                 !test->existsAs<edm::ParameterSet>("f19") && !test->existsAs<edm::ParameterSet>("f20") &&
                 !test->existsAs<edm::ParameterSet>("f21") && !test->existsAs<edm::ParameterSet>("f22") &&
                 !test->existsAs<edm::ParameterSet>("f23") && !test->existsAs<edm::ParameterSet>("f24") &&
                 !test->existsAs<edm::ParameterSet>("f25") && !test->existsAs<edm::ParameterSet>("f26") &&
                 !test->existsAs<edm::ParameterSet>("f27") && !test->existsAs<edm::ParameterSet>("f28") &&
                 !test->existsAs<edm::ParameterSet>("f29") && !test->existsAs<edm::ParameterSet>("f30"));

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("a1") && !test->existsAs<edm::ParameterSet>("a2") &&
                 !test->existsAs<edm::ParameterSet>("a3") && !test->existsAs<edm::ParameterSet>("a4") &&
                 !test->existsAs<edm::ParameterSet>("a5") && !test->existsAs<edm::ParameterSet>("a6") &&
                 !test->existsAs<edm::ParameterSet>("a7") && !test->existsAs<edm::ParameterSet>("a8") &&
                 !test->existsAs<edm::ParameterSet>("a9") && !test->existsAs<edm::ParameterSet>("a10"));

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("o1") && !test->existsAs<edm::ParameterSet>("o2") &&
                 !test->existsAs<edm::ParameterSet>("o7") && !test->existsAs<edm::ParameterSet>("o8") &&
                 !test->existsAs<edm::ParameterSet>("o9") && !test->existsAs<edm::ParameterSet>("o10"));

  CPPUNIT_ASSERT(test->existsAs<edm::ParameterSet>("ess@ess1") && test->existsAs<edm::ParameterSet>("ess2@") &&
                 !test->existsAs<edm::ParameterSet>("ess@ess3") && !test->existsAs<edm::ParameterSet>("ess4@") &&
                 !test->existsAs<edm::ParameterSet>("ess@ess11") && !test->existsAs<edm::ParameterSet>("ess12@") &&
                 !test->existsAs<edm::ParameterSet>("ess@ess13") && !test->existsAs<edm::ParameterSet>("ess14@") &&
                 !test->existsAs<edm::ParameterSet>("ess@ess15") && !test->existsAs<edm::ParameterSet>("ess16@") &&
                 !test->existsAs<edm::ParameterSet>("ess@ess17") && !test->existsAs<edm::ParameterSet>("ess18@") &&
                 !test->existsAs<edm::ParameterSet>("ess@ess19") && !test->existsAs<edm::ParameterSet>("ess20@") &&
                 !test->existsAs<edm::ParameterSet>("ess@ess21") && !test->existsAs<edm::ParameterSet>("ess22@") &&
                 !test->existsAs<edm::ParameterSet>("ess@ess23") && !test->existsAs<edm::ParameterSet>("ess24@") &&
                 !test->existsAs<edm::ParameterSet>("ess@ess25") && !test->existsAs<edm::ParameterSet>("ess26@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess27") && test->existsAs<edm::ParameterSet>("ess28@"));

  CPPUNIT_ASSERT(test->existsAs<edm::ParameterSet>("esp@esp1") && test->existsAs<edm::ParameterSet>("esp2@") &&
                 !test->existsAs<edm::ParameterSet>("esp@esp3") && !test->existsAs<edm::ParameterSet>("esp4@") &&
                 !test->existsAs<edm::ParameterSet>("esp@esp11") && !test->existsAs<edm::ParameterSet>("esp12@") &&
                 !test->existsAs<edm::ParameterSet>("esp@esp13") && !test->existsAs<edm::ParameterSet>("esp14@") &&
                 !test->existsAs<edm::ParameterSet>("esp@esp15") && !test->existsAs<edm::ParameterSet>("esp16@") &&
                 !test->existsAs<edm::ParameterSet>("esp@esp17") && !test->existsAs<edm::ParameterSet>("esp18@") &&
                 !test->existsAs<edm::ParameterSet>("esp@esp19") && !test->existsAs<edm::ParameterSet>("esp20@") &&
                 !test->existsAs<edm::ParameterSet>("esp@esp21") && !test->existsAs<edm::ParameterSet>("esp22@") &&
                 !test->existsAs<edm::ParameterSet>("esp@esp23") && !test->existsAs<edm::ParameterSet>("esp24@") &&
                 !test->existsAs<edm::ParameterSet>("esp@esp25") && !test->existsAs<edm::ParameterSet>("esp26@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp27") && test->existsAs<edm::ParameterSet>("esp28@"));

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
  CPPUNIT_ASSERT(result);
}

void testmakeprocess::taskTestWithSchedule() {
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

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("m1") && !test->existsAs<edm::ParameterSet>("m2") &&
                 test->existsAs<edm::ParameterSet>("m3") && test->existsAs<edm::ParameterSet>("m4") &&
                 test->existsAs<edm::ParameterSet>("m5") && test->existsAs<edm::ParameterSet>("m6") &&
                 test->existsAs<edm::ParameterSet>("m7") && test->existsAs<edm::ParameterSet>("m8") &&
                 test->existsAs<edm::ParameterSet>("m9") && test->existsAs<edm::ParameterSet>("m10") &&
                 test->existsAs<edm::ParameterSet>("m11") && test->existsAs<edm::ParameterSet>("m12") &&
                 test->existsAs<edm::ParameterSet>("m13") && test->existsAs<edm::ParameterSet>("m14") &&
                 test->existsAs<edm::ParameterSet>("m15") && test->existsAs<edm::ParameterSet>("m16") &&
                 test->existsAs<edm::ParameterSet>("m17") && test->existsAs<edm::ParameterSet>("m18") &&
                 test->existsAs<edm::ParameterSet>("m19") && test->existsAs<edm::ParameterSet>("m20") &&
                 test->existsAs<edm::ParameterSet>("m21") && test->existsAs<edm::ParameterSet>("m22") &&
                 test->existsAs<edm::ParameterSet>("m23") && test->existsAs<edm::ParameterSet>("m24") &&
                 test->existsAs<edm::ParameterSet>("m25") && test->existsAs<edm::ParameterSet>("m26") &&
                 test->existsAs<edm::ParameterSet>("m27") && test->existsAs<edm::ParameterSet>("m28") &&
                 !test->existsAs<edm::ParameterSet>("m29") && !test->existsAs<edm::ParameterSet>("m30"));

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("f1") && !test->existsAs<edm::ParameterSet>("f2") &&
                 test->existsAs<edm::ParameterSet>("f3") && test->existsAs<edm::ParameterSet>("f4") &&
                 test->existsAs<edm::ParameterSet>("f5") && test->existsAs<edm::ParameterSet>("f6") &&
                 test->existsAs<edm::ParameterSet>("f7") && test->existsAs<edm::ParameterSet>("f8") &&
                 test->existsAs<edm::ParameterSet>("f9") && test->existsAs<edm::ParameterSet>("f10") &&
                 test->existsAs<edm::ParameterSet>("f11") && test->existsAs<edm::ParameterSet>("f12") &&
                 test->existsAs<edm::ParameterSet>("f13") && test->existsAs<edm::ParameterSet>("f14") &&
                 test->existsAs<edm::ParameterSet>("f15") && test->existsAs<edm::ParameterSet>("f16") &&
                 test->existsAs<edm::ParameterSet>("f17") && test->existsAs<edm::ParameterSet>("f18") &&
                 test->existsAs<edm::ParameterSet>("f19") && test->existsAs<edm::ParameterSet>("f20") &&
                 test->existsAs<edm::ParameterSet>("f21") && test->existsAs<edm::ParameterSet>("f22") &&
                 test->existsAs<edm::ParameterSet>("f23") && test->existsAs<edm::ParameterSet>("f24") &&
                 test->existsAs<edm::ParameterSet>("f25") && test->existsAs<edm::ParameterSet>("f26") &&
                 test->existsAs<edm::ParameterSet>("f27") && test->existsAs<edm::ParameterSet>("f28") &&
                 !test->existsAs<edm::ParameterSet>("f29") && !test->existsAs<edm::ParameterSet>("f30"));

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("a1") && !test->existsAs<edm::ParameterSet>("a2") &&
                 test->existsAs<edm::ParameterSet>("a3") && test->existsAs<edm::ParameterSet>("a4") &&
                 test->existsAs<edm::ParameterSet>("a5") && test->existsAs<edm::ParameterSet>("a6") &&
                 test->existsAs<edm::ParameterSet>("a7") && test->existsAs<edm::ParameterSet>("a8") &&
                 test->existsAs<edm::ParameterSet>("a9") && test->existsAs<edm::ParameterSet>("a10"));

  CPPUNIT_ASSERT(!test->existsAs<edm::ParameterSet>("o1") && !test->existsAs<edm::ParameterSet>("o2") &&
                 test->existsAs<edm::ParameterSet>("o7") && test->existsAs<edm::ParameterSet>("o8") &&
                 test->existsAs<edm::ParameterSet>("o9") && test->existsAs<edm::ParameterSet>("o10"));

  CPPUNIT_ASSERT(test->existsAs<edm::ParameterSet>("ess@ess1") && test->existsAs<edm::ParameterSet>("ess2@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess3") && test->existsAs<edm::ParameterSet>("ess4@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess11") && test->existsAs<edm::ParameterSet>("ess12@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess13") && test->existsAs<edm::ParameterSet>("ess14@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess15") && test->existsAs<edm::ParameterSet>("ess16@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess17") && test->existsAs<edm::ParameterSet>("ess18@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess19") && test->existsAs<edm::ParameterSet>("ess20@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess21") && test->existsAs<edm::ParameterSet>("ess22@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess23") && test->existsAs<edm::ParameterSet>("ess24@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess25") && test->existsAs<edm::ParameterSet>("ess26@") &&
                 test->existsAs<edm::ParameterSet>("ess@ess27") && test->existsAs<edm::ParameterSet>("ess28@"));

  CPPUNIT_ASSERT(test->existsAs<edm::ParameterSet>("esp@esp1") && test->existsAs<edm::ParameterSet>("esp2@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp3") && test->existsAs<edm::ParameterSet>("esp4@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp11") && test->existsAs<edm::ParameterSet>("esp12@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp13") && test->existsAs<edm::ParameterSet>("esp14@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp15") && test->existsAs<edm::ParameterSet>("esp16@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp17") && test->existsAs<edm::ParameterSet>("esp18@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp19") && test->existsAs<edm::ParameterSet>("esp20@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp21") && test->existsAs<edm::ParameterSet>("esp22@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp23") && test->existsAs<edm::ParameterSet>("esp24@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp25") && test->existsAs<edm::ParameterSet>("esp26@") &&
                 test->existsAs<edm::ParameterSet>("esp@esp27") && test->existsAs<edm::ParameterSet>("esp28@"));

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
  CPPUNIT_ASSERT(result);
}

void testmakeprocess::edmException() {
  {
    char const* const kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "raise cms.EDMException(cms.edm.errors.Configuration,'test message')\n";

    bool exceptionHappened = false;
    try {
      (void)pSet(kTest);
    } catch (edm::Exception const& e) {
      exceptionHappened = true;
      CPPUNIT_ASSERT(e.categoryCode() == edm::errors::Configuration);
    } catch (std::exception const& e) {
      std::cout << "wrong error " << e.what() << std::endl;
      exceptionHappened = true;
      CPPUNIT_ASSERT(false);
    }
    CPPUNIT_ASSERT(exceptionHappened);
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
      CPPUNIT_ASSERT(e.categoryCode() == edm::errors::UnavailableAccelerator);
    } catch (std::exception const& e) {
      std::cout << "wrong error " << e.what() << std::endl;
      exceptionHappened = true;
      CPPUNIT_ASSERT(false);
    }
    CPPUNIT_ASSERT(exceptionHappened);
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
   CPPUNIT_ASSERT(src.getParameter<int>("i") == 1);
   std::string s1 = src.getParameter<std::string>("s1");
   std::string s2 = src.getParameter<std::string>("s2");

   std::cerr << "\nsize of s1 is: " << s1.size();
   std::cerr << "\nsize of s2 is: " << s2.size() << '\n';

   CPPUNIT_ASSERT(s1.size() == 1);
   CPPUNIT_ASSERT(s1[0] == ret);

   CPPUNIT_ASSERT(s2.size() == 2);
   CPPUNIT_ASSERT(s2[0] == backsl);
   CPPUNIT_ASSERT(s2[1] == 'r');
}
*/
