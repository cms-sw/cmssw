/*
 *  makepset_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 5/18/05.
 *
 */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>

#include <iostream>

#include <stdlib.h> // for setenv; <cstdlib> is likely to fail

#include "cppunit/extensions/HelperMacros.h"
#include "boost/shared_ptr.hpp"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"



class testmakepset: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testmakepset);
  CPPUNIT_TEST(typesTest);
  CPPUNIT_TEST(secsourceTest);
  CPPUNIT_TEST(usingBlockTest);
  CPPUNIT_TEST(fileinpathTest);
  CPPUNIT_TEST_SUITE_END();

 public:
  void setUp(){}
  void tearDown(){}
  void typesTest();
  void secsourceTest();
  void usingBlockTest();
  void fileinpathTest();

 private:
  void secsourceAux();
  void usingBlockAux();
  void fileinpathAux();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testmakepset);


void testmakepset::secsourceTest() {
  try { this->secsourceAux(); }
  catch (cms::Exception& x) {
    std::cerr << "testmakepset::secsourceTest() caught a cms::Exception\n";
    std::cerr << x.what() << '\n';
    throw;
  }
  catch (std::exception& x) {
    std::cerr << "testmakepset::secsourceTest() caught a std::exception\n";
    std::cerr << x.what() << '\n';
    throw;
  }
  catch (...) {
    std::cerr << "testmakepset::secsourceTest() caught an unidentified exception\n";
    throw;
  }
}

void testmakepset::secsourceAux() {
  char const* kTest =
  "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('PROD')\n"
  "process.maxEvents = cms.untracked.PSet(\n"
  "    input = cms.untracked.int32(2)\n"
  ")\n"
  "process.source = cms.Source('PoolSource',\n"
  "    fileNames = cms.untracked.vstring('file:main.root')\n"
  ")\n"
  "process.out = cms.OutputModule('PoolOutputModule',\n"
  "    fileName = cms.string('file:CumHits.root')\n"
  ")\n"
  "process.mix = cms.EDFilter('MixingModule',\n"
  "    input = cms.SecSource('PoolSource',\n"
  "        fileNames = cms.untracked.vstring('file:pileup.root')\n"
  "    ),\n"
  "    max_bunch = cms.int32(3),\n"
  "    average_number = cms.double(14.3),\n"
  "    min_bunch = cms.int32(-5),\n"
  "    type = cms.string('fixed')\n"
  ")\n"
  "process.p = cms.Path(process.mix)\n"
  "process.ep = cms.EndPath(process.out)\n";

  std::string config(kTest);

  // Create the ParameterSet object from this configuration string.
  PythonProcessDesc builder(config);
  boost::shared_ptr<edm::ParameterSet> ps = builder.processDesc()->getProcessPSet();

  CPPUNIT_ASSERT(0 != ps.get());

  // Make sure this ParameterSet object has the right contents
  edm::ParameterSet const& mixingModuleParams = ps->getParameterSet("mix");
  edm::ParameterSet const& secondarySourceParams = mixingModuleParams.getParameterSet("input");
  CPPUNIT_ASSERT(secondarySourceParams.getParameter<std::string>("@module_type") == "PoolSource");
  CPPUNIT_ASSERT(secondarySourceParams.getParameter<std::string>("@module_label") == "input");
  CPPUNIT_ASSERT(secondarySourceParams.getUntrackedParameter<std::vector<std::string> >("fileNames")[0] == "file:pileup.root");
}

void testmakepset::usingBlockTest() {
  try { this->usingBlockAux(); }
  catch (cms::Exception& x) {
    std::cerr << "testmakepset::usingBlockTest() caught a cms::Exception\n";
    std::cerr << x.what() << '\n';
    throw;
  }
  catch (...) {
    std::cerr << "testmakepset::usingBlockTest() caught an unidentified exception\n";
    throw;
  }
}

void testmakepset::usingBlockAux() {
  char const* kTest =
  "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('PROD')\n"
  "process.maxEvents = cms.untracked.PSet(\n"
  "    input = cms.untracked.int32(2)\n"
  ")\n"
  "process.source = cms.Source('PoolSource',\n"
  "    fileNames = cms.untracked.vstring('file:main.root')\n"
  ")\n"
  "process.b = cms.PSet(\n"
  "    s = cms.string('original'),\n"
  "    r = cms.double(1.5)\n"
  ")\n"
  "process.m1 = cms.EDFilter('Class1',\n"
  "    process.b,\n"
  "    i = cms.int32(1)\n"
  ")\n"
  "process.m2 = cms.EDFilter('Class2',\n"
  "    process.b,\n"
  "    i = cms.int32(2),\n"
  "    j = cms.int32(3),\n"
  "    u = cms.uint64(1011),\n"
  "    l = cms.int64(101010)\n"
  ")\n";


  std::string config(kTest);
  // Create the ParameterSet object from this configuration string.
  PythonProcessDesc builder(config);
  boost::shared_ptr<edm::ParameterSet> ps = builder.processDesc()->getProcessPSet();

  CPPUNIT_ASSERT(0 != ps.get());

  // Make sure this ParameterSet object has the right contents
  edm::ParameterSet const& m1Params = ps->getParameterSet("m1");
  edm::ParameterSet const& m2Params = ps->getParameterSet("m2");
  CPPUNIT_ASSERT(m1Params.getParameter<int>("i") == 1);
  CPPUNIT_ASSERT(m2Params.getParameter<int>("i") == 2);
  CPPUNIT_ASSERT(m2Params.getParameter<int>("j") == 3);
  CPPUNIT_ASSERT(m2Params.getParameter<long long>("l") == 101010);
  CPPUNIT_ASSERT(m2Params.getParameter<unsigned long long>("u") == 1011);

  CPPUNIT_ASSERT(m1Params.getParameter<std::string>("s") == "original");
  CPPUNIT_ASSERT(m2Params.getParameter<std::string>("s") == "original");

  CPPUNIT_ASSERT(m1Params.getParameter<double>("r") == 1.5);
  CPPUNIT_ASSERT(m2Params.getParameter<double>("r") == 1.5);
}

void testmakepset::fileinpathTest() {
  try { this->fileinpathAux(); }
  catch (cms::Exception& x) {
    std::cerr << "testmakepset::fileinpathTest() caught a cms::Exception\n";
    std::cerr << x.what() << '\n';
    throw;
  }
  catch (...) {
    std::cerr << "testmakepset::fileinpathTest() caught an unidentified exception\n";
    throw;
  }
}

void testmakepset::fileinpathAux() {
  char const* kTest =
  "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('PROD')\n"
  "process.main = cms.PSet(\n"
  "    topo = cms.FileInPath('Geometry/TrackerSimData/data/trackersens.xml'),\n"
  "    fip = cms.FileInPath('FWCore/ParameterSet/python/Config.py'),\n"
  "    ufip = cms.untracked.FileInPath('FWCore/ParameterSet/python/Types.py'),\n"
  "    extraneous = cms.int32(12)\n"
  ")\n"
  "process.source = cms.Source('DummySource')\n";

  std::string config(kTest);

  // Create the ParameterSet object from this configuration string.
  PythonProcessDesc builder(config);
  boost::shared_ptr<edm::ParameterSet> ps = builder.processDesc()->getProcessPSet();
  CPPUNIT_ASSERT(0 != ps.get());

  edm::ParameterSet const& innerps = ps->getParameterSet("main");
  edm::FileInPath fip  = innerps.getParameter<edm::FileInPath>("fip");
  edm::FileInPath ufip = innerps.getUntrackedParameter<edm::FileInPath>("ufip");
  CPPUNIT_ASSERT(innerps.existsAs<int>("extraneous"));
  CPPUNIT_ASSERT(!innerps.existsAs<int>("absent"));
  char *releaseBase = getenv("CMSSW_RELEASE_BASE");
  bool localArea = (releaseBase != 0 && strlen(releaseBase) != 0);
  if(localArea) {
    CPPUNIT_ASSERT(fip.isLocal() == true);
  }
  CPPUNIT_ASSERT(fip.relativePath()  == "FWCore/ParameterSet/python/Config.py");
  CPPUNIT_ASSERT(ufip.relativePath() == "FWCore/ParameterSet/python/Types.py");
  std::string fullpath = fip.fullPath();
  std::cerr << "fullPath is: " << fip.fullPath() << std::endl;
  std::cerr << "copy of fullPath is: " << fullpath << std::endl;

  CPPUNIT_ASSERT(!fullpath.empty());

  std::string tmpout = fullpath.substr(0, fullpath.find("FWCore/ParameterSet/python/Config.py")) + "tmp.py";

  edm::FileInPath topo = innerps.getParameter<edm::FileInPath>("topo");
  CPPUNIT_ASSERT(topo.isLocal() == false);
  CPPUNIT_ASSERT(topo.relativePath() == "Geometry/TrackerSimData/data/trackersens.xml");
  fullpath = topo.fullPath();
  CPPUNIT_ASSERT(!fullpath.empty());

  std::vector<edm::FileInPath> v(1);
  CPPUNIT_ASSERT(innerps.getAllFileInPaths(v) == 3);

  CPPUNIT_ASSERT(v.size() == 4);
  CPPUNIT_ASSERT(std::count(v.begin(), v.end(), fip) == 1);
  CPPUNIT_ASSERT(std::count(v.begin(), v.end(), topo) == 1);

  edm::ParameterSet empty;
  v.clear();
  CPPUNIT_ASSERT(empty.getAllFileInPaths(v) == 0);
  CPPUNIT_ASSERT(v.empty());

  // This last test checks that a FileInPath parameter can be read
  // successfully even if the associated file no longer exists.
  std::ofstream out(tmpout.c_str());
  CPPUNIT_ASSERT(!(!out));

  char const* kTest2 =
  "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('PROD')\n"
  "process.main = cms.PSet(\n"
  "    fip2 = cms.FileInPath('tmp.py')\n"
  ")\n"
  "process.source = cms.Source('DummySource')\n";

  std::string config2(kTest2);
  // Create the ParameterSet object from this configuration string.
  PythonProcessDesc builder2(config2);
  unlink(tmpout.c_str());
  boost::shared_ptr<edm::ParameterSet> ps2 = builder2.processDesc()->getProcessPSet();

  CPPUNIT_ASSERT(0 != ps2.get());

  edm::ParameterSet const& innerps2 = ps2->getParameterSet("main");
  edm::FileInPath fip2 = innerps2.getParameter<edm::FileInPath>("fip2");
  if (localArea) {
    CPPUNIT_ASSERT(fip2.isLocal() == true);
  }
  CPPUNIT_ASSERT(fip2.relativePath() == "tmp.py");
  std::string fullpath2 = fip2.fullPath();
  std::cerr << "fullPath is: " << fip2.fullPath() << std::endl;
  std::cerr << "copy of fullPath is: " << fullpath2 << std::endl;
  CPPUNIT_ASSERT(!fullpath2.empty());
}

void testmakepset::typesTest() {
   //vbool vb = {true, false};
   char const* kTest =
  "import FWCore.ParameterSet.Config as cms\n"
  "process = cms.Process('t')\n"
  "process.p = cms.PSet(\n"
  "    input2 = cms.InputTag('Label2','Instance2'),\n"
  "    sb3 = cms.string('    '),\n"
  "    input1 = cms.InputTag('Label1','Instance1'),\n"
  "    input6 = cms.InputTag('source'),\n"
  "    #justasbig = cms.double(inf),\n"
  "    input4 = cms.InputTag('Label4','Instance4','Process4'),\n"
  "    input3 = cms.untracked.InputTag('Label3','Instance3'),\n"
  "    h2 = cms.uint32(255),\n"
  "    vi = cms.vint32(1, -2),\n"
  "    input8 = cms.string('deprecatedString:tag'),\n"
  "    h1 = cms.int32(74),\n"
  "    vs = cms.vstring('','1', \n"
  "        '2', \n"
  "        'a', 'XXX'),\n"
  "    vs2 = cms.vstring(), vs3 = cms.vstring(''),\n"
  "    sb2 = cms.string(''),\n"
  "    input7 = cms.InputTag('source','sink'),\n"
  "    ps = cms.PSet(\n"
  "        b2 = cms.untracked.bool(True)\n"
  "    ),\n"
  "    input5 = cms.InputTag('Label5','','Process5'),\n"
  "    h3 = cms.untracked.uint32(3487559679),\n"
  "    input = cms.InputTag('Label'),\n"
  "    vps = cms.VPSet(cms.PSet(\n"
  "        b3 = cms.bool(False)\n"
  "    )),\n"
  "    #indebt = cms.double(-inf),\n"
  "    #big = cms.double(inf),\n"
  "    vinput = cms.VInputTag(cms.InputTag('l1','i1'), cms.InputTag('l2'), cms.InputTag('l3','i3','p3'), cms.InputTag('l4','','p4'), cms.InputTag('source'), \n"
  "        cms.InputTag('source','sink')),\n"
  "    ui = cms.uint32(1),\n"
  "    eventID = cms.EventID(1, 0, 1),\n"
  "    b = cms.untracked.bool(True),\n"
  "    d = cms.double(1.0),\n"
  "    i = cms.int32(1),\n"
  "    vui = cms.vuint32(1, 2, 1, 255),\n"
  "    s = cms.string('this string'),\n"
  "    sb1 = cms.string(''),\n"
  "    emptyString = cms.untracked.string(''),\n"
  "    vEventID = cms.VEventID('1:1', '2:2','3:3'),\n"
  "    lumi = cms.LuminosityBlockID(55, 65),\n"
  "    vlumis = cms.VLuminosityBlockID('75:85', '95:105'),\n"
  "    einput1 = cms.ESInputTag(),\n"
  "    einput2 = cms.ESInputTag(data='blah'),\n"
  "    einput3 = cms.ESInputTag('ESProd'),\n"
  "    einput4 = cms.ESInputTag('ESProd','something'),\n"
  "    einput5 = cms.ESInputTag('ESProd:something'),\n"
  "    veinput1 = cms.VESInputTag(),\n"
  "    veinput2 = cms.VESInputTag(cms.ESInputTag(data='blah'),cms.ESInputTag('ESProd'))\n"
  ")\n"

     ;

   std::string config2(kTest);
   // Create the ParameterSet object from this configuration string.
   PythonProcessDesc builder2(config2);
   boost::shared_ptr<edm::ParameterSet> ps2 = builder2.processDesc()->getProcessPSet();
   edm::ParameterSet const& test = ps2->getParameterSet("p");

   CPPUNIT_ASSERT(1 == test.getParameter<int>("i"));
   CPPUNIT_ASSERT(test.retrieve("i").isTracked());
   CPPUNIT_ASSERT(1 == test.getParameter<unsigned int>("ui"));
   CPPUNIT_ASSERT(1 == test.getParameter<double>("d"));
   //CPPUNIT_ASSERT(100000. < test.getParameter<double>("big"));
   //CPPUNIT_ASSERT(100000. < test.getParameter<double>("justasbig"));
   //CPPUNIT_ASSERT(-1000000. > test.getParameter<double>("indebt"));

   // test hex numbers
   CPPUNIT_ASSERT(74 == test.getParameter<int>("h1"));
   CPPUNIT_ASSERT(255 == test.getParameter<unsigned int>("h2"));
   CPPUNIT_ASSERT(3487559679U == test.getUntrackedParameter<unsigned int>("h3"));

   //std::cout << test.getParameter<std::string>("s") << std::endl;
   CPPUNIT_ASSERT("this string" == test.getParameter<std::string>("s"));
   //std::cout <<"blank string using single quotes returns \""<<test.getParameter<std::string>("sb1")<<"\""<<std::endl;
   //std::cout <<"blank string using double quotes returns \""<<test.getParameter<std::string>("sb2")<<"\""<<std::endl;
   CPPUNIT_ASSERT("" == test.getParameter<std::string>("sb1"));
   CPPUNIT_ASSERT("" == test.getUntrackedParameter<std::string>("emptyString", "default"));
   CPPUNIT_ASSERT("" == test.getParameter<std::string>("sb2"));
   CPPUNIT_ASSERT(4  == test.getParameter<std::string>("sb3").size());
   std::vector<std::string> vs = test.getParameter<std::vector<std::string> >("vs");
   int vssize = vs.size();
   //FIXME doesn't do spaces right
   edm::Entry e(test.retrieve("vs"));
   CPPUNIT_ASSERT(5 == vssize);
   CPPUNIT_ASSERT(vssize && "" == vs[0]);
   CPPUNIT_ASSERT(vssize >1 && "1" == vs[1]);
   CPPUNIT_ASSERT(vssize >1 && "a" == vs[3]);
   //std::cout <<"\""<<test.getParameter<std::vector<std::string> >("vs")[0]<<"\" \""<<test.getParameter<std::vector<std::string> >("vs")[1]<<"\" \""
   //<<test.getParameter<std::vector<std::string> >("vs")[2]<<"\""<<std::endl;
   vs = test.getParameter<std::vector<std::string> >("vs2");
   CPPUNIT_ASSERT(vs.size() == 0);
   vs = test.getParameter<std::vector<std::string> >("vs3");
   CPPUNIT_ASSERT(vs.size() == 1);
   CPPUNIT_ASSERT(vs[0] == "");

   static unsigned int const vuia[] = {1,2,1,255};
   static std::vector<unsigned int> const vui(vuia, vuia+sizeof(vuia)/sizeof(unsigned int));
   CPPUNIT_ASSERT(vui == test.getParameter<std::vector<unsigned int> >("vui"));

   static int const via[] = {1,-2};
   static std::vector<int> const vi(via, via+sizeof(vuia)/sizeof(unsigned int));
   test.getParameter<std::vector<int> >("vi");
   CPPUNIT_ASSERT(true == test.getUntrackedParameter<bool>("b", false));
   CPPUNIT_ASSERT(test.retrieve("vi").isTracked());
   //test.getParameter<std::vector<bool> >("vb");
   edm::ParameterSet const& ps = test.getParameterSet("ps");
   CPPUNIT_ASSERT(true == ps.getUntrackedParameter<bool>("b2", false));
   std::vector<edm::ParameterSet> const& vps = test.getParameterSetVector("vps");
   CPPUNIT_ASSERT(1 == vps.size());
   CPPUNIT_ASSERT(false == vps.front().getParameter<bool>("b3"));

   // InputTag
   edm::InputTag inputProduct  = test.getParameter<edm::InputTag>("input");
   edm::InputTag inputProduct1 = test.getParameter<edm::InputTag>("input1");
   edm::InputTag inputProduct2 = test.getParameter<edm::InputTag>("input2");
   edm::InputTag inputProduct3 = test.getUntrackedParameter<edm::InputTag>("input3");
   edm::InputTag inputProduct4 = test.getParameter<edm::InputTag>("input4");
   edm::InputTag inputProduct5 = test.getParameter<edm::InputTag>("input5");
   edm::InputTag inputProduct6 = test.getParameter<edm::InputTag>("input6");
   edm::InputTag inputProduct7 = test.getParameter<edm::InputTag>("input7");
   edm::InputTag inputProduct8 = test.getParameter<edm::InputTag>("input8");

   //edm::OutputTag outputProduct = test.getParameter<edm::OutputTag>("output");

   CPPUNIT_ASSERT("Label"    == inputProduct.label());
   CPPUNIT_ASSERT("Label1"    == inputProduct1.label());
   CPPUNIT_ASSERT("Label2"    == inputProduct2.label());
   CPPUNIT_ASSERT("Instance2" == inputProduct2.instance());
   CPPUNIT_ASSERT("Label3"    == inputProduct3.label());
   CPPUNIT_ASSERT("Instance3" == inputProduct3.instance());
   CPPUNIT_ASSERT("Label4" == inputProduct4.label());
   CPPUNIT_ASSERT("Instance4" == inputProduct4.instance());
   CPPUNIT_ASSERT("Process4" == inputProduct4.process());
   CPPUNIT_ASSERT("Label5" == inputProduct5.label());
   CPPUNIT_ASSERT("" == inputProduct5.instance());
   CPPUNIT_ASSERT("Process5" == inputProduct5.process());
   CPPUNIT_ASSERT("source" == inputProduct6.label());
   CPPUNIT_ASSERT("source" == inputProduct7.label());
   CPPUNIT_ASSERT("deprecatedString" == inputProduct8.label());


   // vector of InputTags

   std::vector<edm::InputTag> vtags = test.getParameter<std::vector<edm::InputTag> >("vinput");
   CPPUNIT_ASSERT("l1" == vtags[0].label());
   CPPUNIT_ASSERT("i1" == vtags[0].instance());
   CPPUNIT_ASSERT("l2" == vtags[1].label());
   CPPUNIT_ASSERT("l3" == vtags[2].label());
   CPPUNIT_ASSERT("i3" == vtags[2].instance());
   CPPUNIT_ASSERT("p3" == vtags[2].process());
   CPPUNIT_ASSERT("l4" == vtags[3].label());
   CPPUNIT_ASSERT(""   == vtags[3].instance());
   CPPUNIT_ASSERT("p4" == vtags[3].process());
   CPPUNIT_ASSERT("source" == vtags[4].label());
   CPPUNIT_ASSERT("source" == vtags[5].label());

   // ESInputTag
   edm::ESInputTag einput1 = test.getParameter<edm::ESInputTag>("einput1");
   edm::ESInputTag einput2 = test.getParameter<edm::ESInputTag>("einput2");
   edm::ESInputTag einput3 = test.getParameter<edm::ESInputTag>("einput3");
   edm::ESInputTag einput4 = test.getParameter<edm::ESInputTag>("einput4");
   edm::ESInputTag einput5 = test.getParameter<edm::ESInputTag>("einput5");
   CPPUNIT_ASSERT("" == einput1.module());
   CPPUNIT_ASSERT("" == einput1.data());
   CPPUNIT_ASSERT("" == einput2.module());
   CPPUNIT_ASSERT("blah" == einput2.data());
   CPPUNIT_ASSERT("ESProd" == einput3.module());
   CPPUNIT_ASSERT("" == einput3.data());
   CPPUNIT_ASSERT("ESProd" == einput4.module());
   CPPUNIT_ASSERT("something" == einput4.data());
   CPPUNIT_ASSERT("ESProd" == einput5.module());
   CPPUNIT_ASSERT("something" == einput5.data());

   std::vector<edm::ESInputTag> veinput1 = test.getParameter<std::vector<edm::ESInputTag> >("veinput1");
   std::vector<edm::ESInputTag> veinput2 = test.getParameter<std::vector<edm::ESInputTag> >("veinput2");
   CPPUNIT_ASSERT(0 == veinput1.size());
   CPPUNIT_ASSERT(2 == veinput2.size());
   CPPUNIT_ASSERT("" == veinput2[0].module());
   CPPUNIT_ASSERT("blah" == veinput2[0].data());
   CPPUNIT_ASSERT("ESProd" == veinput2[1].module());
   CPPUNIT_ASSERT("" == veinput2[1].data());

   edm::EventID eventID = test.getParameter<edm::EventID>("eventID");
   std::vector<edm::EventID> vEventID = test.getParameter<std::vector<edm::EventID> >("vEventID");
   CPPUNIT_ASSERT(1 == eventID.run());
   CPPUNIT_ASSERT(1 == eventID.event());
   CPPUNIT_ASSERT(1 == vEventID[0].run());
   CPPUNIT_ASSERT(1 == vEventID[0].event());
   CPPUNIT_ASSERT(3 == vEventID[2].run());
   CPPUNIT_ASSERT(3 == vEventID[2].event());

   edm::LuminosityBlockID lumi = test.getParameter<edm::LuminosityBlockID >("lumi");
   CPPUNIT_ASSERT(55 == lumi.run());
   CPPUNIT_ASSERT(65 == lumi.luminosityBlock());
   std::vector<edm::LuminosityBlockID> vlumis = test.getParameter<std::vector<edm::LuminosityBlockID> >("vlumis");
   CPPUNIT_ASSERT(vlumis.size() == 2);
   CPPUNIT_ASSERT(vlumis[0].run() == 75);
   CPPUNIT_ASSERT(vlumis[0].luminosityBlock() == 85);
   CPPUNIT_ASSERT(vlumis[1].run() == 95);
   CPPUNIT_ASSERT(vlumis[1].luminosityBlock() == 105);


   //CPPUNIT_ASSERT("Label2" == outputProduct.label());
   //CPPUNIT_ASSERT(""       == outputProduct.instance());
   //CPPUNIT_ASSERT("Alias2" == outputProduct.alias());
   //BOOST_CHECK_THROW(makePSet(*nodeList), std::runtime_error);
}

