/*
 *  makepset_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 5/18/05.
 *
 */

#include "catch2/catch_all.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/resolveSymbolicLinks.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

#include <memory>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdlib.h>  // for setenv; <cstdlib> is likely to fail
#include <string>
#include <unistd.h>
#include <filesystem>

// Helper functions
static void secsourceAux();
static void usingBlockAux();
static void fileinpathAux();

TEST_CASE("MakePset", "[PythonParameterSet]") {
  SECTION("secsource") {
    try {
      secsourceAux();
    } catch (cms::Exception& x) {
      std::cerr << "secsource test caught a cms::Exception\n";
      std::cerr << x.what() << '\n';
      throw;
    } catch (std::exception& x) {
      std::cerr << "secsource test caught a std::exception\n";
      std::cerr << x.what() << '\n';
      throw;
    } catch (...) {
      std::cerr << "secsource test caught an unidentified exception\n";
      throw;
    }
  }

  SECTION("usingBlock") {
    try {
      usingBlockAux();
    } catch (cms::Exception& x) {
      std::cerr << "usingBlock test caught a cms::Exception\n";
      std::cerr << x.what() << '\n';
      throw;
    } catch (...) {
      std::cerr << "usingBlock test caught an unidentified exception\n";
      throw;
    }
  }

  SECTION("fileinpath") {
    try {
      fileinpathAux();
    } catch (cms::Exception& x) {
      std::cerr << "fileinpath test caught a cms::Exception\n";
      std::cerr << x.what() << '\n';
      throw;
    } catch (...) {
      std::cerr << "fileinpath test caught an unidentified exception\n";
      throw;
    }
  }

  SECTION("types") {
    //vbool vb = {true, false};
    char const* kTest =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('t')\n"
        "process.p = cms.PSet(\n"
        "    input2 = cms.InputTag('Label2','Instance2'),\n"
        "    sb3 = cms.string('    '),\n"
        "    input1 = cms.InputTag('Label1','Instance1'),\n"
        "    input6 = cms.InputTag('source'),\n"
        "    ##justasbig = cms.double(inf),\n"
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
        "    ##indebt = cms.double(-inf),\n"
        "    ##big = cms.double(inf),\n"
        "    vinput = cms.VInputTag(cms.InputTag('l1','i1'), cms.InputTag('l2'), cms.InputTag('l3','i3','p3'), "
        "cms.InputTag('l4','','p4'), cms.InputTag('source'), \n"
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
        "    einput3 = cms.ESInputTag('ESProd:'),\n"
        "    einput4 = cms.ESInputTag('ESProd','something'),\n"
        "    einput5 = cms.ESInputTag('ESProd:something'),\n"
        "    veinput1 = cms.VESInputTag(),\n"
        "    veinput2 = cms.VESInputTag(cms.ESInputTag(data='blah'),cms.ESInputTag('ESProd:'))\n"
        ")\n"

        ;

    std::string config2(kTest);
    // Create the ParameterSet object from this configuration string.
    PyBind11ProcessDesc builder2(config2, false);
    std::shared_ptr<edm::ParameterSet> ps2 = builder2.parameterSet();
    edm::ParameterSet const& test = ps2->getParameterSet("p");

    REQUIRE(1 == test.getParameter<int>("i"));
    REQUIRE(test.retrieve("i").isTracked());
    REQUIRE(1 == test.getParameter<unsigned int>("ui"));
    REQUIRE(1 == test.getParameter<double>("d"));
    //REQUIRE(100000. < test.getParameter<double>("big"));
    //REQUIRE(100000. < test.getParameter<double>("justasbig"));
    //REQUIRE(-1000000. > test.getParameter<double>("indebt"));

    // test hex numbers
    REQUIRE(74 == test.getParameter<int>("h1"));
    REQUIRE(255 == test.getParameter<unsigned int>("h2"));
    REQUIRE(3487559679U == test.getUntrackedParameter<unsigned int>("h3"));

    REQUIRE("this string" == test.getParameter<std::string>("s"));
    REQUIRE("" == test.getParameter<std::string>("sb1"));
    REQUIRE("" == test.getUntrackedParameter<std::string>("emptyString", "default"));
    REQUIRE("" == test.getParameter<std::string>("sb2"));
    REQUIRE(4 == test.getParameter<std::string>("sb3").size());
    std::vector<std::string> vs = test.getParameter<std::vector<std::string> >("vs");
    int vssize = vs.size();
    //FIXME doesn't do spaces right
    edm::Entry e(test.retrieve("vs"));
    REQUIRE(5 == vssize);
    REQUIRE(vssize);
    REQUIRE("" == vs[0]);
    REQUIRE(vssize > 1);
    REQUIRE("1" == vs[1]);
    REQUIRE(vssize > 1);
    REQUIRE("a" == vs[3]);
    vs = test.getParameter<std::vector<std::string> >("vs2");
    REQUIRE(vs.size() == 0);
    vs = test.getParameter<std::vector<std::string> >("vs3");
    REQUIRE(vs.size() == 1);
    REQUIRE(vs[0] == "");

    static unsigned int const vuia[] = {1, 2, 1, 255};
    static std::vector<unsigned int> const vui(vuia, vuia + sizeof(vuia) / sizeof(unsigned int));
    REQUIRE(vui == test.getParameter<std::vector<unsigned int> >("vui"));

    static int const via[] = {1, -2};
    static std::vector<int> const vi(via, via + sizeof(via) / sizeof(int));
    test.getParameter<std::vector<int> >("vi");
    REQUIRE(true == test.getUntrackedParameter<bool>("b", false));
    REQUIRE(test.retrieve("vi").isTracked());
    //test.getParameter<std::vector<bool> >("vb");
    edm::ParameterSet const& ps = test.getParameterSet("ps");
    REQUIRE(true == ps.getUntrackedParameter<bool>("b2", false));
    std::vector<edm::ParameterSet> const& vps = test.getParameterSetVector("vps");
    REQUIRE(1 == vps.size());
    REQUIRE(false == vps.front().getParameter<bool>("b3"));

    // InputTag
    edm::InputTag inputProduct = test.getParameter<edm::InputTag>("input");
    edm::InputTag inputProduct1 = test.getParameter<edm::InputTag>("input1");
    edm::InputTag inputProduct2 = test.getParameter<edm::InputTag>("input2");
    edm::InputTag inputProduct3 = test.getUntrackedParameter<edm::InputTag>("input3");
    edm::InputTag inputProduct4 = test.getParameter<edm::InputTag>("input4");
    edm::InputTag inputProduct5 = test.getParameter<edm::InputTag>("input5");
    edm::InputTag inputProduct6 = test.getParameter<edm::InputTag>("input6");
    edm::InputTag inputProduct7 = test.getParameter<edm::InputTag>("input7");
    edm::InputTag inputProduct8 = test.getParameter<edm::InputTag>("input8");

    //edm::OutputTag outputProduct = test.getParameter<edm::OutputTag>("output");

    REQUIRE("Label" == inputProduct.label());
    REQUIRE("Label1" == inputProduct1.label());
    REQUIRE("Label2" == inputProduct2.label());
    REQUIRE("Instance2" == inputProduct2.instance());
    REQUIRE("Label3" == inputProduct3.label());
    REQUIRE("Instance3" == inputProduct3.instance());
    REQUIRE("Label4" == inputProduct4.label());
    REQUIRE("Instance4" == inputProduct4.instance());
    REQUIRE("Process4" == inputProduct4.process());
    REQUIRE("Label5" == inputProduct5.label());
    REQUIRE("" == inputProduct5.instance());
    REQUIRE("Process5" == inputProduct5.process());
    REQUIRE("source" == inputProduct6.label());
    REQUIRE("source" == inputProduct7.label());
    REQUIRE("deprecatedString" == inputProduct8.label());

    // vector of InputTags

    std::vector<edm::InputTag> vtags = test.getParameter<std::vector<edm::InputTag> >("vinput");
    REQUIRE("l1" == vtags[0].label());
    REQUIRE("i1" == vtags[0].instance());
    REQUIRE("l2" == vtags[1].label());
    REQUIRE("l3" == vtags[2].label());
    REQUIRE("i3" == vtags[2].instance());
    REQUIRE("p3" == vtags[2].process());
    REQUIRE("l4" == vtags[3].label());
    REQUIRE("" == vtags[3].instance());
    REQUIRE("p4" == vtags[3].process());
    REQUIRE("source" == vtags[4].label());
    REQUIRE("source" == vtags[5].label());

    // ESInputTag
    edm::ESInputTag einput1 = test.getParameter<edm::ESInputTag>("einput1");
    edm::ESInputTag einput2 = test.getParameter<edm::ESInputTag>("einput2");
    edm::ESInputTag einput3 = test.getParameter<edm::ESInputTag>("einput3");
    edm::ESInputTag einput4 = test.getParameter<edm::ESInputTag>("einput4");
    edm::ESInputTag einput5 = test.getParameter<edm::ESInputTag>("einput5");
    REQUIRE("" == einput1.module());
    REQUIRE("" == einput1.data());
    REQUIRE("" == einput2.module());
    REQUIRE("blah" == einput2.data());
    REQUIRE("ESProd" == einput3.module());
    REQUIRE("" == einput3.data());
    REQUIRE("ESProd" == einput4.module());
    REQUIRE("something" == einput4.data());
    REQUIRE("ESProd" == einput5.module());
    REQUIRE("something" == einput5.data());

    std::vector<edm::ESInputTag> veinput1 = test.getParameter<std::vector<edm::ESInputTag> >("veinput1");
    std::vector<edm::ESInputTag> veinput2 = test.getParameter<std::vector<edm::ESInputTag> >("veinput2");
    REQUIRE(0 == veinput1.size());
    REQUIRE(2 == veinput2.size());
    REQUIRE("" == veinput2[0].module());
    REQUIRE("blah" == veinput2[0].data());
    REQUIRE("ESProd" == veinput2[1].module());
    REQUIRE("" == veinput2[1].data());

    edm::EventID eventID = test.getParameter<edm::EventID>("eventID");
    std::vector<edm::EventID> vEventID = test.getParameter<std::vector<edm::EventID> >("vEventID");
    REQUIRE(1 == eventID.run());
    REQUIRE(1 == eventID.event());
    REQUIRE(1 == vEventID[0].run());
    REQUIRE(1 == vEventID[0].event());
    REQUIRE(3 == vEventID[2].run());
    REQUIRE(3 == vEventID[2].event());

    edm::LuminosityBlockID lumi = test.getParameter<edm::LuminosityBlockID>("lumi");
    REQUIRE(55 == lumi.run());
    REQUIRE(65 == lumi.luminosityBlock());
    std::vector<edm::LuminosityBlockID> vlumis = test.getParameter<std::vector<edm::LuminosityBlockID> >("vlumis");
    REQUIRE(vlumis.size() == 2);
    REQUIRE(vlumis[0].run() == 75);
    REQUIRE(vlumis[0].luminosityBlock() == 85);
    REQUIRE(vlumis[1].run() == 95);
    REQUIRE(vlumis[1].luminosityBlock() == 105);

    //REQUIRE("Label2" == outputProduct.label());
    //REQUIRE(""       == outputProduct.instance());
    //REQUIRE("Alias2" == outputProduct.alias());
    //BOOST_CHECK_THROW(makePSet(*nodeList), std::runtime_error);
  }
}

static void secsourceAux() {
  char const* kTest =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('PROD')\n"
      "process.maxEvents = cms.untracked.PSet(\n"
      "    input = cms.untracked.int32(2)\n"
      ")\n"
      "process.source = cms.Source('FileBasedSource',\n"
      "    fileNames = cms.untracked.vstring('file:main.root')\n"
      ")\n"
      "process.out = cms.OutputModule('FileBasedOutputModule',\n"
      "    fileName = cms.string('file:CumHits.root')\n"
      ")\n"
      "process.mix = cms.EDFilter('MixingModule',\n"
      "    input = cms.SecSource('EmbeddedFileBasedSource',\n"
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
  PyBind11ProcessDesc builder(config, false);
  std::shared_ptr<edm::ParameterSet> ps = builder.parameterSet();

  REQUIRE(nullptr != ps.get());

  // Make sure this ParameterSet object has the right contents
  edm::ParameterSet const& mixingModuleParams = ps->getParameterSet("mix");
  edm::ParameterSet const& secondarySourceParams = mixingModuleParams.getParameterSet("input");
  REQUIRE(secondarySourceParams.getParameter<std::string>("@module_type") == "EmbeddedFileBasedSource");
  REQUIRE(secondarySourceParams.getParameter<std::string>("@module_label") == "input");
  REQUIRE(secondarySourceParams.getUntrackedParameter<std::vector<std::string> >("fileNames")[0] == "file:pileup.root");
}

static void usingBlockAux() {
  char const* kTest =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('PROD')\n"
      "process.maxEvents = cms.untracked.PSet(\n"
      "    input = cms.untracked.int32(2)\n"
      ")\n"
      "process.source = cms.Source('FileBasedSource',\n"
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
      ")\n"
      "process.p = cms.Path(process.m1+process.m2)\n";

  std::string config(kTest);
  // Create the ParameterSet object from this configuration string.
  PyBind11ProcessDesc builder(config, false);
  std::shared_ptr<edm::ParameterSet> ps = builder.parameterSet();
  REQUIRE(nullptr != ps.get());

  // Make sure this ParameterSet object has the right contents
  edm::ParameterSet const& m1Params = ps->getParameterSet("m1");
  edm::ParameterSet const& m2Params = ps->getParameterSet("m2");
  REQUIRE(m1Params.getParameter<int>("i") == 1);
  REQUIRE(m2Params.getParameter<int>("i") == 2);
  REQUIRE(m2Params.getParameter<int>("j") == 3);
  REQUIRE(m2Params.getParameter<long long>("l") == 101010);
  REQUIRE(m2Params.getParameter<unsigned long long>("u") == 1011);

  REQUIRE(m1Params.getParameter<std::string>("s") == "original");
  REQUIRE(m2Params.getParameter<std::string>("s") == "original");

  REQUIRE(m1Params.getParameter<double>("r") == 1.5);
  REQUIRE(m2Params.getParameter<double>("r") == 1.5);
}

static void fileinpathAux() {
  char const* kTest =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('PROD')\n"
      "process.main = cms.PSet(\n"
      "    topo = cms.FileInPath('Geometry/TrackerSimData/data/trackersens.xml'),\n"
      "    fip = cms.FileInPath('FWCore/PythonParameterSet/test/fip.txt'),\n"
      "    ufip = cms.untracked.FileInPath('FWCore/PythonParameterSet/test/ufip.txt'),\n"
      "    extraneous = cms.int32(12)\n"
      ")\n"
      "process.source = cms.Source('EmptySource')\n";

  std::string config(kTest);
  std::string tmpout;
  bool localArea = false;
  // Create the ParameterSet object from this configuration string.
  {
    PyBind11ProcessDesc builder(config, false);
    std::shared_ptr<edm::ParameterSet> ps = builder.parameterSet();
    REQUIRE(nullptr != ps.get());

    edm::ParameterSet const& innerps = ps->getParameterSet("main");
    edm::FileInPath fip = innerps.getParameter<edm::FileInPath>("fip");
    edm::FileInPath ufip = innerps.getUntrackedParameter<edm::FileInPath>("ufip");
    REQUIRE(innerps.existsAs<int>("extraneous"));
    REQUIRE(!innerps.existsAs<int>("absent"));
    char* releaseBase = std::getenv("CMSSW_RELEASE_BASE");
    char* localBase = std::getenv("CMSSW_BASE");
    localArea = (releaseBase != nullptr && strlen(releaseBase) != 0 && strcmp(releaseBase, localBase));
    if (localArea) {
      // Need to account for possible symbolic links
      std::string const src("/src");
      std::string release = releaseBase + src;
      std::string local = localBase + src;
      edm::resolveSymbolicLinks(release);
      edm::resolveSymbolicLinks(local);
      localArea = (local != release);
    }

    if (localArea) {
      REQUIRE(fip.location() == edm::FileInPath::Local);
    }
    REQUIRE(fip.relativePath() == "FWCore/PythonParameterSet/test/fip.txt");
    REQUIRE(ufip.relativePath() == "FWCore/PythonParameterSet/test/ufip.txt");
    std::string fullpath = fip.fullPath();
    std::cerr << "fullPath is: " << fip.fullPath() << std::endl;
    std::cerr << "copy of fullPath is: " << fullpath << std::endl;

    REQUIRE(!fullpath.empty());

    tmpout = fullpath.substr(0, fullpath.find("src/FWCore/PythonParameterSet/test/fip.txt")) + "tmp/tmp.py";

    edm::FileInPath topo = innerps.getParameter<edm::FileInPath>("topo");
    // if the file is local, then just disable this check as then it is expected to fail
    {
      std::string const src("/src");
      std::string local = localBase + src;
      std::string localFile = local + "/Geometry/TrackerSimData/data/trackersens.xml";
      if (!std::filesystem::exists(localFile))
        REQUIRE(topo.location() != edm::FileInPath::Local);
      else
        std::cerr << "Disabling test against local path for trackersens.xml as package is checked out in this test"
                  << std::endl;
    }
    REQUIRE(topo.relativePath() == "Geometry/TrackerSimData/data/trackersens.xml");
    fullpath = topo.fullPath();
    REQUIRE(!fullpath.empty());

    std::vector<edm::FileInPath> v(1);
    REQUIRE(innerps.getAllFileInPaths(v) == 3);

    REQUIRE(v.size() == 4);
    REQUIRE(std::count(v.begin(), v.end(), fip) == 1);
    REQUIRE(std::count(v.begin(), v.end(), topo) == 1);

    edm::ParameterSet empty;
    v.clear();
    REQUIRE(empty.getAllFileInPaths(v) == 0);
    REQUIRE(v.empty());
  }
  // This last test checks that a FileInPath parameter can be read
  // successfully even if the associated file no longer exists.
  std::ofstream out(tmpout.c_str());
  REQUIRE(!(!out));

  char const* kTest2 =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('PROD')\n"
      "process.main = cms.PSet(\n"
      "    fip2 = cms.FileInPath('tmp.py')\n"
      ")\n"
      "process.source = cms.Source('EmptySource')\n";

  std::string config2(kTest2);
  // Create the ParameterSet object from this configuration string.
  PyBind11ProcessDesc builder2(config2, false);
  unlink(tmpout.c_str());
  std::shared_ptr<edm::ParameterSet> ps2 = builder2.parameterSet();

  REQUIRE(nullptr != ps2.get());

  edm::ParameterSet const& innerps2 = ps2->getParameterSet("main");
  edm::FileInPath fip2 = innerps2.getParameter<edm::FileInPath>("fip2");
  if (localArea) {
    REQUIRE(fip2.location() == edm::FileInPath::Local);
  }
  REQUIRE(fip2.relativePath() == "tmp.py");
  std::string fullpath2 = fip2.fullPath();
  std::cerr << "fullPath is: " << fip2.fullPath() << std::endl;
  std::cerr << "copy of fullPath is: " << fullpath2 << std::endl;
  REQUIRE(!fullpath2.empty());
}
