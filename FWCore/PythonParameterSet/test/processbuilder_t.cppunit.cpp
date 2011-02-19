/**

@file : processbuilder_t.cpp

@brief test suit for process building and schedule validation

*/

#include <cppunit/extensions/HelperMacros.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/PythonParameterSet/interface/PythonProcessDesc.h>
#include "FWCore/Utilities/interface/EDMException.h"

#include "boost/shared_ptr.hpp"

#include <vector>
#include <string>
#include <iostream>

class testProcessDesc: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testProcessDesc);

  CPPUNIT_TEST(trivialPathTest);
  CPPUNIT_TEST(simplePathTest);
  CPPUNIT_TEST(sequenceSubstitutionTest);
  CPPUNIT_TEST(attriggertest);
  CPPUNIT_TEST(nestedSequenceSubstitutionTest);
  CPPUNIT_TEST(sequenceSubstitutionTest2);
  CPPUNIT_TEST(sequenceSubstitutionTest3);
  CPPUNIT_TEST(multiplePathsTest);
  // python throws some different exception
  //CPPUNIT_TEST_EXCEPTION(inconsistentPathTest,edm::Exception);
  //CPPUNIT_TEST_EXCEPTION(inconsistentMultiplePathTest,edm::Exception);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void trivialPathTest();
  void simplePathTest();
  void sequenceSubstitutionTest();
  void attriggertest();
  void nestedSequenceSubstitutionTest();
  void sequenceSubstitutionTest2();
  void sequenceSubstitutionTest3();
  void multiplePathsTest();
  void inconsistentPathTest();
  void inconsistentMultiplePathTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testProcessDesc);

void testProcessDesc::trivialPathTest() {
  std::string str =
  "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('X')\n"
  "process.a = cms.EDFilter('A',\n"
  "    p = cms.int32(3)\n"
  ")\n"
  "process.b = cms.EDProducer('B')\n"
  "process.c = cms.EDProducer('C')\n"
  "process.p = cms.Path(process.a*process.b*process.c)\n";

  boost::shared_ptr<edm::ParameterSet> test = PythonProcessDesc(str).processDesc();

  typedef std::vector<std::string> Strs;

  Strs s = (*test).getParameter<std::vector<std::string> >("p");
  CPPUNIT_ASSERT(s[0]=="a");
  //CPPUNIT_ASSERT(b->getDependencies("a")=="");
}

void testProcessDesc::simplePathTest() {

  std::string str =
  "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('X')\n"
  "process.a = cms.EDFilter('A',\n"
  "    p = cms.int32(3)\n"
  ")\n"
  "process.b = cms.EDFilter('A',\n"
  "    p = cms.int32(3)\n"
  ")\n"
  "process.c = cms.EDFilter('A',\n"
  "    p = cms.int32(3)\n"
  ")\n"
  "process.p = cms.Path(process.a*process.b*process.c)\n";

  boost::shared_ptr<edm::ParameterSet> test = PythonProcessDesc(str).processDesc();

  typedef std::vector<std::string> Strs;

  Strs s = (*test).getParameter<std::vector<std::string> >("p");
  CPPUNIT_ASSERT(s[0]=="a");
  CPPUNIT_ASSERT(s[1]=="b");
  CPPUNIT_ASSERT(s[2]=="c");

  //CPPUNIT_ASSERT (b->getDependencies("a")=="");
  //CPPUNIT_ASSERT (b->getDependencies("b")=="a,");
  //CPPUNIT_ASSERT (b->getDependencies("c")=="a,b,");
}

void testProcessDesc:: attriggertest () {

  std::string str =
  "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('test')\n"
  "process.cone1 = cms.EDFilter('PhonyConeJet',\n"
  "    i = cms.int32(5)\n"
  ")\n"
  "process.cone2 = cms.EDFilter('PhonyConeJet',\n"
  "    i = cms.int32(7)\n"
  ")\n"
  "process.somejet1 = cms.EDFilter('PhonyJet',\n"
  "    i = cms.int32(7)\n"
  ")\n"
  "process.somejet2 = cms.EDFilter('PhonyJet',\n"
  "    i = cms.int32(7)\n"
  ")\n"
  "process.jtanalyzer = cms.EDFilter('PhonyConeJet',\n"
  "    i = cms.int32(7)\n"
  ")\n"
  "process.output = cms.OutputModule('OutputModule')\n"
  "process.cones = cms.Sequence(process.cone1*process.cone2)\n"
  "process.jets = cms.Sequence(process.somejet1*process.somejet2)\n"
  "process.path1 = cms.Path(process.cones*process.jets*process.jtanalyzer)\n"
  "process.epath = cms.EndPath(process.output)\n";

  try {
  boost::shared_ptr<edm::ParameterSet> test = PythonProcessDesc(str).processDesc();

  typedef std::vector<std::string> Strs;

  edm::ParameterSet const& trig_pset =
   (*test).getParameterSet("@trigger_paths");
  Strs tnames = trig_pset.getParameter<Strs>("@trigger_paths");
  Strs enames = (*test).getParameter<Strs>("@end_paths");

  std::cerr << trig_pset.toString() << "\n";

  CPPUNIT_ASSERT(tnames[0]=="path1");
  CPPUNIT_ASSERT(enames[0]=="epath");

  // see if the auto-schedule is correct
  Strs schedule = (*test).getParameter<Strs>("@paths");
  CPPUNIT_ASSERT(schedule.size() == 2);
  CPPUNIT_ASSERT(schedule[0] == "path1");
  CPPUNIT_ASSERT(schedule[1] == "epath");

  }
  catch (cms::Exception& exc) {
    std::cerr << "Got an cms::Exception: " << exc.what() << "\n";
    throw;
  }
  catch (std::exception& exc) {
    std::cerr << "Got an std::exception: " << exc.what() << "\n";
    throw;
  }
  catch (...) {
    std::cerr << "Got an unknown exception: " << "\n";
    throw;
  }
}

void testProcessDesc:: sequenceSubstitutionTest () {

  std::string str = "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('test')\n"
  "process.cone1 = cms.EDFilter('PhonyConeJet',\n"
  "    i = cms.int32(5)\n"
  ")\n"
  "process.cone2 = cms.EDFilter('PhonyConeJet',\n"
  "    i = cms.int32(7)\n"
  ")\n"
  "process.somejet1 = cms.EDFilter('PhonyJet',\n"
  "    i = cms.int32(7)\n"
  ")\n"
  "process.somejet2 = cms.EDFilter('PhonyJet',\n"
  "    i = cms.int32(7)\n"
  ")\n"
  "process.jtanalyzer = cms.EDFilter('PhonyConeJet',\n"
  "    i = cms.int32(7)\n"
  ")\n"
  "process.cones = cms.Sequence(process.cone1*process.cone2)\n"
  "process.jets = cms.Sequence(process.somejet1*process.somejet2)\n"
  "process.path1 = cms.Path(process.cones*process.jets*process.jtanalyzer)\n";

  boost::shared_ptr<edm::ParameterSet> test = PythonProcessDesc(str).processDesc();

  typedef std::vector<std::string> Strs;

  Strs s = (*test).getParameter<std::vector<std::string> >("path1");
  CPPUNIT_ASSERT(s[0]=="cone1");
  CPPUNIT_ASSERT(s[1]=="cone2");
  CPPUNIT_ASSERT(s[2]=="somejet1");
  CPPUNIT_ASSERT(s[3]=="somejet2");
  CPPUNIT_ASSERT(s[4]=="jtanalyzer");

  //CPPUNIT_ASSERT (b->getDependencies("cone1")=="");
  //CPPUNIT_ASSERT (b->getDependencies("cone2")=="cone1,");
  //CPPUNIT_ASSERT (b->getDependencies("somejet1")=="cone1,cone2,");
  //CPPUNIT_ASSERT (b->getDependencies("somejet2")=="cone1,cone2,somejet1,");
  //CPPUNIT_ASSERT (b->getDependencies("jtanalyzer")=="cone1,cone2,somejet1,somejet2,");
}

void testProcessDesc::nestedSequenceSubstitutionTest() {

  std::string str = "import FWCore.ParameterSet.Config as cms\n"
   "process = cms.Process('test')\n"
   "process.a = cms.EDProducer('PhonyConeJet', i = cms.int32(5))\n"
   "process.b = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.c = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
   "process.d = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
   "process.s1 = cms.Sequence( process.a+ process.b)\n"
   "process.s2 = cms.Sequence(process.s1+ process.c)\n"
   "process.path1 = cms.Path(process.s2+process.d)\n";
  boost::shared_ptr<edm::ParameterSet> test = PythonProcessDesc(str).processDesc();

  typedef std::vector<std::string> Strs;

  Strs s = (*test).getParameter<std::vector<std::string> >("path1");
  CPPUNIT_ASSERT(s[0]=="a");
  CPPUNIT_ASSERT(s[1]=="b");
  CPPUNIT_ASSERT(s[2]=="c");
  CPPUNIT_ASSERT(s[3]=="d");

  //CPPUNIT_ASSERT (b.getDependencies("a")=="");
  //CPPUNIT_ASSERT (b.getDependencies("b")=="a,");
  //CPPUNIT_ASSERT (b.getDependencies("c")=="a,b,");
  //CPPUNIT_ASSERT (b.getDependencies("d")=="a,b,c,");
}

void testProcessDesc::sequenceSubstitutionTest2() {

  std::string str = "import FWCore.ParameterSet.Config as cms\n"
   "process = cms.Process('test')\n"
   "process.cone1 = cms.EDProducer('PhonyConeJet', i = cms.int32(5))\n"
   "process.cone2 = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.cone3 = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.somejet1 = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
   "process.somejet2 = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
   "process.jtanalyzer = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.cones = cms.Sequence(process.cone1+ process.cone2+ process.cone3)\n"
   "process.jets = cms.Sequence(process.somejet1+ process.somejet2)\n"
   "process.path1 = cms.Path(process.cones+process.jets+ process.jtanalyzer)\n";

  boost::shared_ptr<edm::ParameterSet> test = PythonProcessDesc(str).processDesc();

  typedef std::vector<std::string> Strs;

  Strs s = (*test).getParameter<std::vector<std::string> >("path1");
  CPPUNIT_ASSERT(s[0]=="cone1");
  CPPUNIT_ASSERT(s[1]=="cone2");
  CPPUNIT_ASSERT(s[2]=="cone3");
  CPPUNIT_ASSERT(s[3]=="somejet1");
  CPPUNIT_ASSERT(s[4]=="somejet2");
  CPPUNIT_ASSERT(s[5]=="jtanalyzer");

  //CPPUNIT_ASSERT (b.getDependencies("cone1")=="");
  //CPPUNIT_ASSERT (b.getDependencies("cone2")=="cone1,");
  //CPPUNIT_ASSERT (b.getDependencies("cone3")=="cone1,cone2,");
  //CPPUNIT_ASSERT (b.getDependencies("somejet1")=="cone1,cone2,cone3,");
  //CPPUNIT_ASSERT (b.getDependencies("somejet2")=="cone1,cone2,cone3,somejet1,");
  //CPPUNIT_ASSERT (b.getDependencies("jtanalyzer")=="cone1,cone2,cone3,somejet1,somejet2,");
}

void testProcessDesc::sequenceSubstitutionTest3() {

   std::string str = "import FWCore.ParameterSet.Config as cms\n"
   "process = cms.Process('test')\n"
   "process.a = cms.EDProducer('PhonyConeJet', i = cms.int32(5))\n"
   "process.b = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.c = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.aa = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
   "process.bb = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
   "process.cc = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.dd = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.aaa = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.bbb = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.ccc = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.ddd = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.eee = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.last = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"

   "process.s1 = cms.Sequence(process.a* process.b* process.c)\n"
   "process.s2 = cms.Sequence(process.aa*process.bb*cms.ignore(process.cc)*process.dd)\n"
   "process.s3 = cms.Sequence(process.aaa*process.bbb*~process.ccc*process.ddd*process.eee)\n"
   "process.path1 = cms.Path(process.s1+process.s3+process.s2+process.last)\n";

  boost::shared_ptr<edm::ParameterSet> test = PythonProcessDesc(str).processDesc();

  typedef std::vector<std::string> Strs;

  Strs s = (*test).getParameter<std::vector<std::string> >("path1");
  CPPUNIT_ASSERT(s[0]=="a");
  CPPUNIT_ASSERT(s[1]=="b");
  CPPUNIT_ASSERT(s[2]=="c");
  CPPUNIT_ASSERT(s[3]=="aaa");
  CPPUNIT_ASSERT(s[4]=="bbb");
  CPPUNIT_ASSERT(s[5]=="!ccc");
  CPPUNIT_ASSERT(s[6]=="ddd");
  CPPUNIT_ASSERT(s[7]=="eee");
  CPPUNIT_ASSERT(s[8]=="aa");
  CPPUNIT_ASSERT(s[9]=="bb");
  CPPUNIT_ASSERT(s[10]=="-cc");
  CPPUNIT_ASSERT(s[11]=="dd");
  CPPUNIT_ASSERT(s[12]=="last");

  //CPPUNIT_ASSERT (b.getDependencies("a")=="");
  //CPPUNIT_ASSERT (b.getDependencies("b")=="a,");
  //CPPUNIT_ASSERT (b.getDependencies("c")=="a,b,");
  //CPPUNIT_ASSERT (b.getDependencies("aaa")=="a,b,c,");
  //CPPUNIT_ASSERT (b.getDependencies("bbb")=="a,aaa,b,c,");
  //CPPUNIT_ASSERT (b.getDependencies("ccc")=="a,aaa,b,bbb,c,");
  //CPPUNIT_ASSERT (b.getDependencies("ddd")=="a,aaa,b,bbb,c,ccc,");
  //CPPUNIT_ASSERT (b.getDependencies("eee")=="a,aaa,b,bbb,c,ccc,ddd,");
  //CPPUNIT_ASSERT (b.getDependencies("aa")=="a,aaa,b,bbb,c,ccc,ddd,eee,");
  //CPPUNIT_ASSERT (b.getDependencies("bb")=="a,aa,aaa,b,bbb,c,ccc,ddd,eee,");
  //CPPUNIT_ASSERT (b.getDependencies("cc")=="a,aa,aaa,b,bb,bbb,c,ccc,ddd,eee,");
  //CPPUNIT_ASSERT (b.getDependencies("dd")=="a,aa,aaa,b,bb,bbb,c,cc,ccc,ddd,eee,");
  //CPPUNIT_ASSERT (b.getDependencies("last")=="a,aa,aaa,b,bb,bbb,c,cc,ccc,dd,ddd,eee,");

}

void testProcessDesc::multiplePathsTest() {

  std::string str = "import FWCore.ParameterSet.Config as cms\n"
    "process = cms.Process('test')\n"
    "process.cone1 = cms.EDProducer('PhonyConeJet', i = cms.int32(5))\n"
    "process.cone2 = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
    "process.cone3 = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
    "process.somejet1 = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
    "process.somejet2 = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
    "process.jtanalyzer = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
    "process.anotherjtanalyzer = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
    "process.cones = cms.Sequence(process.cone1* process.cone2* process.cone3)\n"
    "process.jets = cms.Sequence(process.somejet1* process.somejet2)\n"
    "process.path1 = cms.Path(process.cones+ process.jtanalyzer)\n"
    "process.path2 = cms.Path(process.jets+ process.anotherjtanalyzer)\n"
    "process.schedule = cms.Schedule(process.path2, process.path1)\n";

  boost::shared_ptr<edm::ParameterSet> test = PythonProcessDesc(str).processDesc();

  typedef std::vector<std::string> Strs;

  Strs s = (*test).getParameter<std::vector<std::string> >("path1");
  CPPUNIT_ASSERT(s[0]=="cone1");
  CPPUNIT_ASSERT(s[1]=="cone2");
  CPPUNIT_ASSERT(s[2]=="cone3");
  CPPUNIT_ASSERT(s[3]=="jtanalyzer");

  //CPPUNIT_ASSERT (b.getDependencies("cone1")=="");
  //CPPUNIT_ASSERT (b.getDependencies("cone2")=="cone1,");
  //CPPUNIT_ASSERT (b.getDependencies("cone3")=="cone1,cone2,");
  //CPPUNIT_ASSERT (b.getDependencies("jtanalyzer")=="cone1,cone2,cone3,");

  s = (*test).getParameter<std::vector<std::string> >("path2");
  CPPUNIT_ASSERT(s[0]=="somejet1");
  CPPUNIT_ASSERT(s[1]=="somejet2");
  CPPUNIT_ASSERT(s[2]=="anotherjtanalyzer");

  //CPPUNIT_ASSERT (b.getDependencies("somejet1")=="");
  //CPPUNIT_ASSERT (b.getDependencies("somejet2")=="somejet1,");
  //CPPUNIT_ASSERT (b.getDependencies("anotherjtanalyzer")=="somejet1,somejet2,");

  Strs schedule = (*test).getParameter<std::vector<std::string> >("@paths");

  CPPUNIT_ASSERT (schedule.size() == 2);
  CPPUNIT_ASSERT (schedule[0] == "path2");
  CPPUNIT_ASSERT (schedule[1] == "path1");
}

void testProcessDesc::inconsistentPathTest() {

  std::string str = "import FWCore.ParameterSet.Config as cms\n"
   "process = cms.Process('test')\n"
   "process.a = cms.EDProducer('PhonyConeJet', i = cms.int32(5))\n"
   "process.b = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.c = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
   "process.path1 = cms.Path((process.a*process.b)+ (process.c*process.b))\n";
  boost::shared_ptr<edm::ParameterSet> test = PythonProcessDesc(str).processDesc();
}

void testProcessDesc::inconsistentMultiplePathTest() {

   std::string str = "import FWCore.ParameterSet.Config as cms\n"
   "process = cms.Process('test')\n"
   "process.cone1 = cms.EDProducer('PhonyConeJet', i = cms.int32(5))\n"
   "process.cone2 = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.somejet1 = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
   "process.somejet2 = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
   "process.jtanalyzer = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
   "process.cones = cms.Sequence(process.cone1+ process.cone2)\n"
   "process.jets = cms.Sequence(process.somejet1* process.somejet2)\n"
   "process.path1 = cms.Path(process.cones*process.jtanalyzer)\n"
   "process.path2 = cms.Path(process.jets*process.jtanalyzer)\n";

  boost::shared_ptr<edm::ParameterSet> test = PythonProcessDesc(str).processDesc();
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
