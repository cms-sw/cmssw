#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/TestProcessor/interface/TestSourceProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/OtherThing.h"
#include <vector>
#include <filesystem>
#include "catch.hpp"

static constexpr auto s_tag = "[RNTupleOutputSource]";

namespace {
  std::string setOutputFile(std::string const& iConfig, std::string const& iFileName) {
    using namespace std::string_literals;
    return iConfig + "\nprocess.out.fileName = '"s + iFileName + "'\n";
  }

  std::string setInputFile(std::string const& iConfig, std::string const& iFileName) {
    using namespace std::string_literals;
    return iConfig + "\nprocess.source.fileNames = ['file:"s + iFileName + "']\n";
  }
}  // namespace
TEST_CASE("Tests of RNTupleOuput -> RNTupleSource", s_tag) {
  const std::string baseOutConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.out = cms.OutputModule('RNTupleOutputModule',
                      fileName = cms.untracked.string('')
)
process.add_(cms.Service("InitRootHandlers"))
process.add_(cms.Service("JobReportService"))

process.moduleToTest(process.out)
)_"};

  const std::string baseSourceConfig{
      R"_(from FWCore.TestProcessor.TestSourceProcess import *
process = TestSourceProcess()
process.source = cms.Source("RNTupleSource", fileNames = cms.untracked.vstring())
process.add_(cms.Service("InitRootHandlers"))
process.add_(cms.Service("SiteLocalConfigService"))
process.add_(cms.Service("JobReportService"))
    )_"};

  SECTION("OneEmptyEvent") {
    const std::string fileName = "one_event.rntpl";
    {
      auto configString = setOutputFile(baseOutConfig, fileName);

      edm::test::TestProcessor::Config config{configString};

      edm::test::TestProcessor tester(config);
      tester.test();
    }
    {
      auto config = setInputFile(baseSourceConfig, fileName);
      edm::test::TestSourceProcessor tester(config);

      {
        auto n = tester.findNextTransition();
        REQUIRE(n == edm::InputSource::ItemType::IsFile);
        auto f = tester.openFile();
        REQUIRE(bool(f));
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsRun);
        auto r = tester.readRun();
        REQUIRE(r.run() == 1);
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsLumi);
        auto r = tester.readLuminosityBlock();
        REQUIRE(r.run() == 1);
        REQUIRE(r.luminosityBlock() == 1);
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsEvent);
        auto r = tester.readEvent();
        REQUIRE(r.run() == 1);
        REQUIRE(r.luminosityBlock() == 1);
        REQUIRE(r.event() == 1);
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsStop);
      }
    }
    std::filesystem::remove(fileName);
  }

  SECTION("EventWithThing") {
    const std::string fileName = "thing.rntpl";
    {
      auto configString = setOutputFile(baseOutConfig, fileName);

      edm::test::TestProcessor::Config config{configString};
      auto thingToken = config.produces<std::vector<edmtest::Thing>>("thing");

      edm::test::TestProcessor tester(config);
      tester.test(std::make_pair(thingToken, std::make_unique<std::vector<edmtest::Thing>>(1, edmtest::Thing{1})));
    }
    {
      auto config = setInputFile(baseSourceConfig, fileName);
      edm::test::TestSourceProcessor tester(config);

      {
        auto n = tester.findNextTransition();
        REQUIRE(n == edm::InputSource::ItemType::IsFile);
        auto f = tester.openFile();
        REQUIRE(bool(f));
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsRun);
        auto r = tester.readRun();
        REQUIRE(r.run() == 1);
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsLumi);
        auto r = tester.readLuminosityBlock();
        REQUIRE(r.run() == 1);
        REQUIRE(r.luminosityBlock() == 1);
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsEvent);
        auto r = tester.readEvent();
        REQUIRE(r.run() == 1);
        REQUIRE(r.luminosityBlock() == 1);
        REQUIRE(r.event() == 1);
        auto v = r.get<std::vector<edmtest::Thing>>("thing", "", "TEST");
        REQUIRE(v.isValid());
        REQUIRE(v->size() == 1);
        REQUIRE((*v)[0].a == 1);
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsStop);
      }
    }
    std::filesystem::remove(fileName);
  }

  SECTION("EventWithRef") {
    const std::string fileName = "ref.rntpl";
    {
      auto configString = setOutputFile(baseOutConfig, fileName) +
                          "process.other = cms.EDProducer('OtherThingProducer', thingTag = cms.InputTag('thing'))\n"
                          "process.moduleToTest(process.out,cms.Task(process.other))\n";

      edm::test::TestProcessor::Config config{configString};

      auto thingToken = config.produces<std::vector<edmtest::Thing>>("thing");

      edm::test::TestProcessor tester(config);
      tester.test(std::make_pair(thingToken, std::make_unique<std::vector<edmtest::Thing>>(1, edmtest::Thing{1})));
    }
    {
      auto config = setInputFile(baseSourceConfig, fileName);
      edm::test::TestSourceProcessor tester(config);

      {
        auto n = tester.findNextTransition();
        REQUIRE(n == edm::InputSource::ItemType::IsFile);
        auto f = tester.openFile();
        REQUIRE(bool(f));
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsRun);
        auto r = tester.readRun();
        REQUIRE(r.run() == 1);
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsLumi);
        auto r = tester.readLuminosityBlock();
        REQUIRE(r.run() == 1);
        REQUIRE(r.luminosityBlock() == 1);
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsEvent);
        auto r = tester.readEvent();
        REQUIRE(r.run() == 1);
        REQUIRE(r.luminosityBlock() == 1);
        REQUIRE(r.event() == 1);
        auto v = r.get<std::vector<edmtest::Thing>>("thing", "", "TEST");
        REQUIRE(v.isValid());
        REQUIRE(v->size() == 1);
        REQUIRE((*v)[0].a == 1);
        auto o = r.get<std::vector<edmtest::OtherThing>>("other", "testUserTag", "TEST");
        REQUIRE(o.isValid());
        REQUIRE(o->size() == 20);
        REQUIRE((*o)[0].a == 0);
        REQUIRE((*o)[1].a == 1);
        REQUIRE((*o)[0].ref.isNonnull());
        REQUIRE((*o)[0].ref->a == 1);
        REQUIRE((*o)[0].refProd.isNonnull());
        REQUIRE((*o)[0].refProd->size() == 1);
        REQUIRE((*o)[0].refVec.size() == 2);
        REQUIRE((*o)[0].refVec[0]->a == 1);
        REQUIRE((*o)[0].ptr.isNonnull());
        REQUIRE((*o)[0].ptr->a == 1);
        REQUIRE((*o)[0].ptrVec.size() == 2);
        REQUIRE((*o)[0].ptrVec[0]->a == 1);
        REQUIRE((*o)[0].refToBaseProd.isNonnull());
        REQUIRE((*o)[0].refToBaseProd->size() == 1);
        REQUIRE((*o)[0].refToBase.isNonnull());
        REQUIRE((*o)[0].refToBase->a == 1);
      }
      {
        auto n = tester.findNextTransition();
        REQUIRE(n.itemType() == edm::InputSource::ItemType::IsStop);
      }
    }
    std::filesystem::remove(fileName);
  }
}
