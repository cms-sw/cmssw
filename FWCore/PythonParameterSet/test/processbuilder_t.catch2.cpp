/**

@file : processbuilder_t.cpp

@brief test suit for process building and schedule validation

*/

#include "catch2/catch_all.hpp"

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h>
#include "FWCore/Utilities/interface/EDMException.h"

#include <memory>

#include <vector>
#include <string>
#include <iostream>

TEST_CASE("ProcessDesc", "[PythonParameterSet]") {
  SECTION("trivialPathTest") {
    std::string str =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('X')\n"
        "process.a = cms.EDFilter('A',\n"
        "    p = cms.int32(3)\n"
        ")\n"
        "process.b = cms.EDProducer('B')\n"
        "process.c = cms.EDProducer('C')\n"
        "process.p = cms.Path(process.a*process.b*process.c)\n";

    std::shared_ptr<edm::ParameterSet> test = PyBind11ProcessDesc(str, false).parameterSet();

    typedef std::vector<std::string> Strs;

    Strs s = (*test).getParameter<std::vector<std::string> >("p");
    REQUIRE(s[0] == "a");
    //REQUIRE(b->getDependencies("a")=="");
  }

  SECTION("simplePathTest") {
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

    std::shared_ptr<edm::ParameterSet> test = PyBind11ProcessDesc(str, false).parameterSet();

    typedef std::vector<std::string> Strs;

    Strs s = (*test).getParameter<std::vector<std::string> >("p");
    REQUIRE(s[0] == "a");
    REQUIRE(s[1] == "b");
    REQUIRE(s[2] == "c");

    //REQUIRE (b->getDependencies("a")=="");
    //REQUIRE (b->getDependencies("b")=="a,");
    //REQUIRE (b->getDependencies("c")=="a,b,");
  }

  SECTION("attriggertest") {
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
      std::shared_ptr<edm::ParameterSet> test = PyBind11ProcessDesc(str, false).parameterSet();

      typedef std::vector<std::string> Strs;

      edm::ParameterSet const& trig_pset = (*test).getParameterSet("@trigger_paths");
      Strs tnames = trig_pset.getParameter<Strs>("@trigger_paths");
      Strs enames = (*test).getParameter<Strs>("@end_paths");

      REQUIRE(tnames[0] == "path1");
      REQUIRE(enames[0] == "epath");

      // see if the auto-schedule is correct
      Strs schedule = (*test).getParameter<Strs>("@paths");
      REQUIRE(schedule.size() == 2);
      REQUIRE(schedule[0] == "path1");
      REQUIRE(schedule[1] == "epath");

    } catch (cms::Exception& exc) {
      std::cerr << "Got an cms::Exception: " << exc.what() << "\n";
      throw;
    } catch (std::exception& exc) {
      std::cerr << "Got an std::exception: " << exc.what() << "\n";
      throw;
    } catch (...) {
      std::cerr << "Got an unknown exception: "
                << "\n";
      throw;
    }
  }

  SECTION("sequenceSubstitutionTest") {
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
        "process.cones = cms.Sequence(process.cone1*process.cone2)\n"
        "process.jets = cms.Sequence(process.somejet1*process.somejet2)\n"
        "process.path1 = cms.Path(process.cones*process.jets*process.jtanalyzer)\n";

    std::shared_ptr<edm::ParameterSet> test = PyBind11ProcessDesc(str, false).parameterSet();

    typedef std::vector<std::string> Strs;

    Strs s = (*test).getParameter<std::vector<std::string> >("path1");
    REQUIRE(s[0] == "cone1");
    REQUIRE(s[1] == "cone2");
    REQUIRE(s[2] == "somejet1");
    REQUIRE(s[3] == "somejet2");
    REQUIRE(s[4] == "jtanalyzer");

    //REQUIRE (b->getDependencies("cone1")=="");
    //REQUIRE (b->getDependencies("cone2")=="cone1,");
    //REQUIRE (b->getDependencies("somejet1")=="cone1,cone2,");
    //REQUIRE (b->getDependencies("somejet2")=="cone1,cone2,somejet1,");
    //REQUIRE (b->getDependencies("jtanalyzer")=="cone1,cone2,somejet1,somejet2,");
  }

  SECTION("nestedSequenceSubstitutionTest") {
    std::string str =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('test')\n"
        "process.a = cms.EDProducer('PhonyConeJet', i = cms.int32(5))\n"
        "process.b = cms.EDProducer('PhonyConeJet', i = cms.int32(7))\n"
        "process.c = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
        "process.d = cms.EDProducer('PhonyJet', i = cms.int32(7))\n"
        "process.s1 = cms.Sequence( process.a+ process.b)\n"
        "process.s2 = cms.Sequence(process.s1+ process.c)\n"
        "process.path1 = cms.Path(process.s2+process.d)\n";
    std::shared_ptr<edm::ParameterSet> test = PyBind11ProcessDesc(str, false).parameterSet();

    typedef std::vector<std::string> Strs;

    Strs s = (*test).getParameter<std::vector<std::string> >("path1");
    REQUIRE(s[0] == "a");
    REQUIRE(s[1] == "b");
    REQUIRE(s[2] == "c");
    REQUIRE(s[3] == "d");

    //REQUIRE (b.getDependencies("a")=="");
    //REQUIRE (b.getDependencies("b")=="a,");
    //REQUIRE (b.getDependencies("c")=="a,b,");
    //REQUIRE (b.getDependencies("d")=="a,b,c,");
  }

  SECTION("sequenceSubstitutionTest2") {
    std::string str =
        "import FWCore.ParameterSet.Config as cms\n"
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

    std::shared_ptr<edm::ParameterSet> test = PyBind11ProcessDesc(str, false).parameterSet();

    typedef std::vector<std::string> Strs;

    Strs s = (*test).getParameter<std::vector<std::string> >("path1");
    REQUIRE(s[0] == "cone1");
    REQUIRE(s[1] == "cone2");
    REQUIRE(s[2] == "cone3");
    REQUIRE(s[3] == "somejet1");
    REQUIRE(s[4] == "somejet2");
    REQUIRE(s[5] == "jtanalyzer");

    //REQUIRE (b.getDependencies("cone1")=="");
    //REQUIRE (b.getDependencies("cone2")=="cone1,");
    //REQUIRE (b.getDependencies("cone3")=="cone1,cone2,");
    //REQUIRE (b.getDependencies("somejet1")=="cone1,cone2,cone3,");
    //REQUIRE (b.getDependencies("somejet2")=="cone1,cone2,cone3,somejet1,");
    //REQUIRE (b.getDependencies("jtanalyzer")=="cone1,cone2,cone3,somejet1,somejet2,");
  }

  SECTION("sequenceSubstitutionTest3") {
    std::string str =
        "import FWCore.ParameterSet.Config as cms\n"
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

    std::shared_ptr<edm::ParameterSet> test = PyBind11ProcessDesc(str, false).parameterSet();

    typedef std::vector<std::string> Strs;

    Strs s = (*test).getParameter<std::vector<std::string> >("path1");
    REQUIRE(s[0] == "a");
    REQUIRE(s[1] == "b");
    REQUIRE(s[2] == "c");
    REQUIRE(s[3] == "aaa");
    REQUIRE(s[4] == "bbb");
    REQUIRE(s[5] == "!ccc");
    REQUIRE(s[6] == "ddd");
    REQUIRE(s[7] == "eee");
    REQUIRE(s[8] == "aa");
    REQUIRE(s[9] == "bb");
    REQUIRE(s[10] == "-cc");
    REQUIRE(s[11] == "dd");
    REQUIRE(s[12] == "last");

    //REQUIRE (b.getDependencies("a")=="");
    //REQUIRE (b.getDependencies("b")=="a,");
    //REQUIRE (b.getDependencies("c")=="a,b,");
    //REQUIRE (b.getDependencies("aaa")=="a,b,c,");
    //REQUIRE (b.getDependencies("bbb")=="a,aaa,b,c,");
    //REQUIRE (b.getDependencies("ccc")=="a,aaa,b,bbb,c,");
    //REQUIRE (b.getDependencies("ddd")=="a,aaa,b,bbb,c,ccc,");
    //REQUIRE (b.getDependencies("eee")=="a,aaa,b,bbb,c,ccc,ddd,");
    //REQUIRE (b.getDependencies("aa")=="a,aaa,b,bbb,c,ccc,ddd,eee,");
    //REQUIRE (b.getDependencies("bb")=="a,aa,aaa,b,bbb,c,ccc,ddd,eee,");
    //REQUIRE (b.getDependencies("cc")=="a,aa,aaa,b,bb,bbb,c,ccc,ddd,eee,");
    //REQUIRE (b.getDependencies("dd")=="a,aa,aaa,b,bb,bbb,c,cc,ccc,ddd,eee,");
    //REQUIRE (b.getDependencies("last")=="a,aa,aaa,b,bb,bbb,c,cc,ccc,dd,ddd,eee,");
  }

  SECTION("multiplePathsTest") {
    std::string str =
        "import FWCore.ParameterSet.Config as cms\n"
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

    std::shared_ptr<edm::ParameterSet> test = PyBind11ProcessDesc(str, false).parameterSet();

    typedef std::vector<std::string> Strs;

    Strs s = (*test).getParameter<std::vector<std::string> >("path1");
    REQUIRE(s[0] == "cone1");
    REQUIRE(s[1] == "cone2");
    REQUIRE(s[2] == "cone3");
    REQUIRE(s[3] == "jtanalyzer");

    //REQUIRE (b.getDependencies("cone1")=="");
    //REQUIRE (b.getDependencies("cone2")=="cone1,");
    //REQUIRE (b.getDependencies("cone3")=="cone1,cone2,");
    //REQUIRE (b.getDependencies("jtanalyzer")=="cone1,cone2,cone3,");

    s = (*test).getParameter<std::vector<std::string> >("path2");
    REQUIRE(s[0] == "somejet1");
    REQUIRE(s[1] == "somejet2");
    REQUIRE(s[2] == "anotherjtanalyzer");

    //REQUIRE (b.getDependencies("somejet1")=="");
    //REQUIRE (b.getDependencies("somejet2")=="somejet1,");
    //REQUIRE (b.getDependencies("anotherjtanalyzer")=="somejet1,somejet2,");

    Strs schedule = (*test).getParameter<std::vector<std::string> >("@paths");

    REQUIRE(schedule.size() == 2);
    REQUIRE(schedule[0] == "path2");
    REQUIRE(schedule[1] == "path1");
  }
}
