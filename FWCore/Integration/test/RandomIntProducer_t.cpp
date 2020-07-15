#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"

static constexpr auto s_tag = "[RandomIntProducer]";

TEST_CASE("Direct", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

process = TestProcess()
process.test = cms.EDProducer('RandomIntProducer')
process.add_(cms.Service("RandomNumberGeneratorService",
                         test = cms.PSet(initialSeed = cms.untracked.uint32(12345))
))
process.moduleToTest(process.test)
)_"};

  edm::test::TestProcessor::Config config{baseConfig};

  SECTION("Base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  SECTION("transitions") {
    edm::test::TestProcessor tester(config);
    {
      auto lumi = tester.testBeginLuminosityBlock(1).get<edmtest::IntProduct>("lumi");
      REQUIRE(lumi->value == 1);
    }
    {
      auto event = tester.test().get<edmtest::IntProduct>();
      REQUIRE(event->value == 6);
    }
    {
      auto event = tester.test().get<edmtest::IntProduct>();
      REQUIRE(event->value == 4);
    }
    {
      auto lumi = tester.testBeginLuminosityBlock(2).get<edmtest::IntProduct>("lumi");
      REQUIRE(lumi->value == 1);
    }
  }
}

TEST_CASE("External", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class RandomIntExternalProcessProducer(cms.EDProducer):
  def __init__(self, prod):
      self.__dict__['_prod'] = prod
      super(cms.EDProducer,self).__init__('TestInterProcessRandomProd')
  def __setattr__(self, name, value):
      setattr(self._prod, name, value)
  def __getattr__(self, name):
      if name =='_prod':
          return self.__dict__['_prod']
      return getattr(self._prod, name)
  def clone(self, **params):
      returnValue = RandomIntExternalProcessProducer.__new__(type(self))
      returnValue.__init__(self._prod.clone())
      return returnValue
  def insertInto(self, parameterSet, myname):
      newpset = parameterSet.newPSet()
      newpset.addString(True, "@module_label", self.moduleLabel_(myname))
      newpset.addString(True, "@module_type", self.type_())
      newpset.addString(True, "@module_edm_type", cms.EDProducer.__name__)
      newpset.addString(True, "@external_type", self._prod.type_())
      newpset.addString(False,"@python_config", self._prod.dumpPython())
      self._prod.insertContentsInto(newpset)
      parameterSet.addPSet(True, self.nameInProcessDesc_(myname), newpset)

process = TestProcess()
_generator = cms.EDProducer('RandomIntProducer')
process.test = RandomIntExternalProcessProducer(_generator)
process.add_(cms.Service("RandomNumberGeneratorService",
                         test = cms.PSet(initialSeed = cms.untracked.uint32(12345))
))
process.moduleToTest(process.test)
)_"};

  edm::test::TestProcessor::Config config{baseConfig};

  SECTION("Base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  SECTION("transitions") {
    edm::test::TestProcessor tester(config);
    {
      auto lumi = tester.testBeginLuminosityBlock(1).get<edmtest::IntProduct>("lumi");
      REQUIRE(lumi->value == 1);
    }
    {
      auto event = tester.test().get<edmtest::IntProduct>();
      REQUIRE(event->value == 6);
    }
    {
      auto event = tester.test().get<edmtest::IntProduct>();
      REQUIRE(event->value == 4);
    }
    {
      auto lumi = tester.testBeginLuminosityBlock(2).get<edmtest::IntProduct>("lumi");
      REQUIRE(lumi->value == 1);
    }
  }
}
