/*
 *  makeprocess_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 5/18/05.
 *  Changed by Viji Sundararajan on 8-Jul-05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 * 
 */


#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"

#include "boost/shared_ptr.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class testmakeprocess: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testmakeprocess);
CPPUNIT_TEST(simpleProcessTest);
CPPUNIT_TEST(usingTest);
CPPUNIT_TEST(pathTest);
CPPUNIT_TEST(moduleTest);
CPPUNIT_TEST(serviceTest);
CPPUNIT_TEST(emptyModuleTest);
//CPPUNIT_TEST(windowsLineEndingTest);
//CPPUNIT_TEST_EXCEPTION(emptyPsetTest,edm::Exception);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void simpleProcessTest();
  void usingTest();
  void pathTest();
  void moduleTest();
  void serviceTest();
  void emptyModuleTest();
  //void windowsLineEndingTest();
private:

  typedef boost::shared_ptr<edm::ProcessDesc> ProcDescPtr;
  ProcDescPtr procDesc(const char * c) {

    //ProcDescPtr result( new edm::ProcessDesc(std::string(c)) );
    ProcDescPtr result = PythonProcessDesc(std::string(c)).processDesc();
    CPPUNIT_ASSERT(result->getProcessPSet()->getParameter<std::string>("@process_name") == "test");
    return result;
  }
  //  void emptyPsetTest();
};



///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testmakeprocess);

void testmakeprocess::simpleProcessTest()
{
   const char* kTest ="import FWCore.ParameterSet.Config as cms\n"
                      "process = cms.Process('test')\n"
                      "dummy =  cms.PSet(b = cms.bool(True))\n";
   ProcDescPtr test = procDesc(kTest);
}

void testmakeprocess::usingTest()
{
   const char* kTest =  "import FWCore.ParameterSet.Config as cms\n"
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
  ")\n";

   
   ProcDescPtr test = procDesc(kTest);

   //CPPUNIT_ASSERT(test->getProcessPSet()->getParameterSet("dummy").getBool("b") == true);   
   CPPUNIT_ASSERT(test->getProcessPSet()->getParameterSet("m1").getParameter<bool>("b") == true);   
   CPPUNIT_ASSERT(test->getProcessPSet()->getParameterSet("m2").getParameter<bool>("d") == true);   
}

void testmakeprocess::pathTest()
{
   const char* kTest =   "import FWCore.ParameterSet.Config as cms\n"
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


   ProcDescPtr test = procDesc(kTest);
   //CPPUNIT_ASSERT(test->pathFragments().size() == 5);

   edm::ParameterSet& myparams = *(test->getProcessPSet());
   myparams.registerIt();
//    std::cout << "ParameterSet looks like:\n";
//    std::cout << myparams.toString() << std::endl;
   std::string rep = myparams.toString();
   edm::ParameterSet copy(rep);
   CPPUNIT_ASSERT(copy == myparams);
}


edm::ParameterSet modulePSet(const std::string& iLabel, const std::string& iType) {
   edm::ParameterSet temp;
   temp.addParameter("s", 1);
   temp.addParameter("@module_label", iLabel);
   temp.addParameter("@module_type", iType);
   return temp;
}

void testmakeprocess::moduleTest()
{
   const char* kTest =  "import FWCore.ParameterSet.Config as cms\n"
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
  ")\n";


   ProcDescPtr test = procDesc(kTest);

   static const edm::ParameterSet kEmpty;
   const edm::ParameterSet kCone(modulePSet("cones", "Module"));
   std::ostringstream out;
   out << kCone.toString() << std::endl;
   out << test->getProcessPSet()->getParameterSet("cones").toString() << std::endl;
   
   const edm::ParameterSet kMainInput(modulePSet("@main_input","InputSource"));
   
   const edm::ParameterSet kNoLabelModule(modulePSet("", "NoLabelModule"));
   const edm::ParameterSet kLabelModule(modulePSet("labeled", "LabelModule"));
   const edm::ParameterSet kNoLabelRetriever(modulePSet("", "NoLabelRetriever"));
   const edm::ParameterSet kLabelRetriever(modulePSet("label", "LabelRetriever"));
   
   CPPUNIT_ASSERT(kEmpty != (test->getProcessPSet()->getParameterSet("cones")));
   CPPUNIT_ASSERT(kCone == test->getProcessPSet()->getParameterSet("cones"));
   
   CPPUNIT_ASSERT(kEmpty != (test->getProcessPSet()->getParameterSet("@main_input")));
   CPPUNIT_ASSERT(kMainInput == (test->getProcessPSet()->getParameterSet("@main_input")));

   CPPUNIT_ASSERT(kEmpty != (test->getProcessPSet()->getParameterSet("NoLabelModule@")));
   CPPUNIT_ASSERT(kNoLabelModule == test->getProcessPSet()->getParameterSet("NoLabelModule@"));
   
   CPPUNIT_ASSERT(kEmpty != (test->getProcessPSet()->getParameterSet("LabelModule@labeled")));
   CPPUNIT_ASSERT(kLabelModule == test->getProcessPSet()->getParameterSet("LabelModule@labeled"));

   CPPUNIT_ASSERT(kEmpty != (test->getProcessPSet()->getParameterSet("NoLabelRetriever@")));
   CPPUNIT_ASSERT(kNoLabelRetriever == test->getProcessPSet()->getParameterSet("NoLabelRetriever@"));

   CPPUNIT_ASSERT(kEmpty != (test->getProcessPSet()->getParameterSet("LabelRetriever@label")));
   CPPUNIT_ASSERT(kLabelRetriever == test->getProcessPSet()->getParameterSet("LabelRetriever@label"));
}

void testmakeprocess::serviceTest()
{
   const char* kTest =
 "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('test')\n"
  "process.XService = cms.Service('XService',\n"
  "    s = cms.int32(1)\n"
  ")\n"
  "process.YService = cms.Service('YService',\n"
  "    s = cms.int32(1)\n"
  ")\n";

   ProcDescPtr test = procDesc(kTest);

   CPPUNIT_ASSERT(test->getServicesPSets()->size() == 2);
   CPPUNIT_ASSERT("XService" == test->getServicesPSets()->at(0).getParameter<std::string>("@service_type"));
   CPPUNIT_ASSERT("YService" == test->getServicesPSets()->at(1).getParameter<std::string>("@service_type"));
}
void testmakeprocess::emptyModuleTest()
{
   const char* kTest =   "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('test')\n"
  "process.thing = cms.EDFilter('XX')\n";

   ProcDescPtr test = procDesc(kTest);

   edm::ParameterSet& myparams = *(test->getProcessPSet());
   myparams.registerIt();
//    std::cout << "ParameterSet looks like:\n";
//    std::cout << myparams.toString() << std::endl;
   std::string rep = myparams.toString();
   edm::ParameterSet copy(rep);
   CPPUNIT_ASSERT(copy == myparams);
}
/*
void testmakeprocess::windowsLineEndingTest()
{

  std::ostringstream oss;
  const char ret = '\r';
  const char nl = '\n';
  const char dquote = '"';
  const char backsl = '\\';

  oss << ret << nl
      << "import FWCore.ParameterSet.Config as cms" << ret << nl
      << "process = cms.Process('test')" << ret << nl
      << "  source = cms.Source('InputSource'," << ret << nl
      << "    i=cms.int32(1)" << ret << nl
      << "    s1 = cms.string(" << dquote << ret << dquote <<  ')' <<ret << nl
      << "    s2 = cms.string(" << dquote << backsl << backsl << 'r' << dquote << ')' << ret << nl
      << "  )" << ret << nl;
  const char* kTest = oss.str().c_str();
  std::cerr << "\n------------------------------\n";
  std::cerr << "s1 will look funky because of the embedded return\n";
  std::cerr << "s2 shows how to get the chars backslash-r into a string\n";
  std::cerr << kTest;
  std::cerr << "\n------------------------------\n";

   ProcDescPtr test = procDesc(kTest);

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
