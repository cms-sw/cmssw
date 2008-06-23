
// $Id: TestMergeResults.cc,v 1.6 2008/05/28 18:52:01 wdd Exp $
//
// Reads some simple test objects in the event, run, and lumi
// principals.  Then checks to see if the values in these
// objects match what we expect.  Intended to be used to
// test the values in a file that has merged run and lumi
// products.
//
// Original Author: David Dagenhart, Fermilab, February 2008

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingWithIsEqual.h"
#include "DataFormats/TestObjects/interface/ThingWithMerge.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <string>
#include <vector>
#include <iostream>

namespace edm {
  class EventSetup;
}

using namespace edm;

namespace edmtest
{

  class TestMergeResults : public edm::EDAnalyzer
  {
  public:

    explicit TestMergeResults(edm::ParameterSet const&);
    virtual ~TestMergeResults();

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
    virtual void beginRun(Run const&, EventSetup const&);
    virtual void endRun(Run const&, EventSetup const&);
    virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&);
    virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&);
    virtual void respondToOpenInputFile(FileBlock const& fb);
    virtual void respondToCloseInputFile(FileBlock const& fb);
    virtual void respondToOpenOutputFiles(FileBlock const& fb);
    virtual void respondToCloseOutputFiles(FileBlock const& fb);
    void endJob();

  private:

    void checkExpectedLumiProducts(unsigned int index,
                                   std::vector<int> const& expectedValues,
                                   InputTag const& tag,
                                   const char* functionName,
                                   LuminosityBlock const& lumi);

    void checkExpectedRunProducts(unsigned int index,
                                  std::vector<int> const& expectedValues,
                                  InputTag const& tag,
                                  const char* functionName,
                                  Run const& run);

    void abortWithMessage(const char* whichFunction, const char* type, edm::InputTag const& tag,
                          int expectedValue, int actualValue) const;

    std::vector<int> default_;
    std::vector<std::string> defaultvstring_;

    std::vector<int> expectedBeginRunProd_;
    std::vector<int> expectedEndRunProd_;
    std::vector<int> expectedBeginLumiProd_;
    std::vector<int> expectedEndLumiProd_;

    std::vector<int> expectedBeginRunNew_;
    std::vector<int> expectedEndRunNew_;
    std::vector<int> expectedBeginLumiNew_;
    std::vector<int> expectedEndLumiNew_;

    std::vector<int> expectedDroppedEvent_;

    int nRespondToOpenInputFile_;
    int nRespondToCloseInputFile_;
    int nRespondToOpenOutputFiles_;
    int nRespondToCloseOutputFiles_;
    int expectedRespondToOpenInputFile_;
    int expectedRespondToCloseInputFile_;
    int expectedRespondToOpenOutputFiles_;
    int expectedRespondToCloseOutputFiles_;

    std::vector<std::string> expectedInputFileNames_;

    bool verbose_;

    unsigned int index0_;
    unsigned int index1_;
    unsigned int index2_;
    unsigned int index3_;
    unsigned int index4_;
    unsigned int index5_;
    unsigned int index6_;
    unsigned int index7_;

    edm::Handle<edmtest::Thing> h_thing;
    edm::Handle<edmtest::ThingWithMerge> h_thingWithMerge;
    edm::Handle<edmtest::ThingWithIsEqual> h_thingWithIsEqual;
  };

  // -----------------------------------------------------------------

  TestMergeResults::TestMergeResults(edm::ParameterSet const& ps):
    default_(),
    defaultvstring_(),
    expectedBeginRunProd_(ps.getUntrackedParameter<std::vector<int> >("expectedBeginRunProd", default_)),
    expectedEndRunProd_(ps.getUntrackedParameter<std::vector<int> >("expectedEndRunProd", default_)),
    expectedBeginLumiProd_(ps.getUntrackedParameter<std::vector<int> >("expectedBeginLumiProd", default_)),
    expectedEndLumiProd_(ps.getUntrackedParameter<std::vector<int> >("expectedEndLumiProd", default_)),

    expectedBeginRunNew_(ps.getUntrackedParameter<std::vector<int> >("expectedBeginRunNew", default_)),
    expectedEndRunNew_(ps.getUntrackedParameter<std::vector<int> >("expectedEndRunNew", default_)),
    expectedBeginLumiNew_(ps.getUntrackedParameter<std::vector<int> >("expectedBeginLumiNew", default_)),
    expectedEndLumiNew_(ps.getUntrackedParameter<std::vector<int> >("expectedEndLumiNew", default_)),

    expectedDroppedEvent_(ps.getUntrackedParameter<std::vector<int> >("expectedDroppedEvent", default_)),

    nRespondToOpenInputFile_(0),
    nRespondToCloseInputFile_(0),
    nRespondToOpenOutputFiles_(0),
    nRespondToCloseOutputFiles_(0),
    expectedRespondToOpenInputFile_(ps.getUntrackedParameter<int>("expectedRespondToOpenInputFile", -1)),
    expectedRespondToCloseInputFile_(ps.getUntrackedParameter<int>("expectedRespondToCloseInputFile", -1)),
    expectedRespondToOpenOutputFiles_(ps.getUntrackedParameter<int>("expectedRespondToOpenOutputFiles", -1)),
    expectedRespondToCloseOutputFiles_(ps.getUntrackedParameter<int>("expectedRespondToCloseOutputFiles", -1)),

    expectedInputFileNames_(ps.getUntrackedParameter<std::vector<std::string> >("expectedInputFileNames", defaultvstring_)),

    verbose_(ps.getUntrackedParameter<bool>("verbose", false)),

    index0_(0),
    index1_(0),
    index2_(0),
    index3_(0),
    index4_(0),
    index5_(0),
    index6_(0),
    index7_(0)
  {
  }

  // -----------------------------------------------------------------

  TestMergeResults::~TestMergeResults()
  {
  }

  // -----------------------------------------------------------------

  void TestMergeResults::analyze(edm::Event const& e,edm::EventSetup const&)
  {
    if (verbose_) edm::LogInfo("TestMergeResults") << "analyze";

    if (expectedDroppedEvent_.size() > 0) {
      edm::InputTag tag1("makeThingToBeDropped", "event", "PROD");
      e.getByLabel(tag1, h_thing);
      assert(h_thing->a == expectedDroppedEvent_[0]);
    }

    Run const& run = e.getRun();
    LuminosityBlock const& lumi = e.getLuminosityBlock();

    edm::InputTag tag0("thingWithMergeProducer", "beginRun", "PROD");
    checkExpectedRunProducts(index0_, expectedBeginRunProd_, tag0, "analyze", run);

    edm::InputTag tag1("thingWithMergeProducer", "beginRun");
    checkExpectedRunProducts(index4_, expectedBeginRunNew_, tag1, "analyze", run);

    edm::InputTag tag2("thingWithMergeProducer", "endRun", "PROD");
    checkExpectedRunProducts(index1_, expectedEndRunProd_, tag2, "analyze", run);

    edm::InputTag tag3("thingWithMergeProducer", "endRun");
    checkExpectedRunProducts(index5_, expectedEndRunNew_, tag3, "analyze", run);

    edm::InputTag tag4("thingWithMergeProducer", "beginLumi", "PROD");
    checkExpectedLumiProducts(index2_, expectedBeginLumiProd_, tag4, "analyze", lumi);

    edm::InputTag tag5("thingWithMergeProducer", "beginLumi");
    checkExpectedLumiProducts(index6_, expectedBeginLumiNew_, tag5, "analyze", lumi);

    edm::InputTag tag6("thingWithMergeProducer", "endLumi", "PROD");
    checkExpectedLumiProducts(index3_, expectedEndLumiProd_, tag6, "analyze", lumi);

    edm::InputTag tag7("thingWithMergeProducer", "endLumi");
    checkExpectedLumiProducts(index7_, expectedEndLumiNew_, tag7, "analyze", lumi);
  }

  void TestMergeResults::beginRun(Run const& run, EventSetup const&) {

    index0_ += 3;
    index4_ += 3;
    index1_ += 3;
    index5_ += 3;

    if (verbose_) edm::LogInfo("TestMergeResults") << "beginRun";

    edm::InputTag tag("thingWithMergeProducer", "beginRun", "PROD");
    checkExpectedRunProducts(index0_, expectedBeginRunProd_, tag, "beginRun", run);

    edm::InputTag tagnew("thingWithMergeProducer", "beginRun");
    checkExpectedRunProducts(index4_, expectedBeginRunNew_, tagnew, "beginRun", run);
  }

  void TestMergeResults::endRun(Run const& run, EventSetup const&) {

    index0_ += 3;
    index4_ += 3;
    index1_ += 3;
    index5_ += 3;

    if (verbose_) edm::LogInfo("TestMergeResults") << "endRun";

    edm::InputTag tag("thingWithMergeProducer", "endRun", "PROD");
    checkExpectedRunProducts(index1_, expectedEndRunProd_, tag, "endRun", run);

    edm::InputTag tagnew("thingWithMergeProducer", "endRun");
    checkExpectedRunProducts(index5_, expectedEndRunNew_, tagnew, "endRun", run);
  }

  void TestMergeResults::beginLuminosityBlock(LuminosityBlock const& lumi, EventSetup const&) {

    index2_ += 3;
    index6_ += 3;
    index3_ += 3;
    index7_ += 3;

    if (verbose_) edm::LogInfo("TestMergeResults") << "beginLuminosityBlock";

    edm::InputTag tag("thingWithMergeProducer", "beginLumi", "PROD");
    checkExpectedLumiProducts(index2_, expectedBeginLumiProd_, tag, "beginLumi", lumi);

    edm::InputTag tagnew("thingWithMergeProducer", "beginLumi");
    checkExpectedLumiProducts(index6_, expectedBeginLumiNew_, tagnew, "beginLumi", lumi);
  }

  void TestMergeResults::endLuminosityBlock(LuminosityBlock const& lumi, EventSetup const&) {

    index2_ += 3;
    index6_ += 3;
    index3_ += 3;
    index7_ += 3;

    if (verbose_) edm::LogInfo("TestMergeResults") << "endLuminosityBlock";

    edm::InputTag tag("thingWithMergeProducer", "endLumi", "PROD");
    checkExpectedLumiProducts(index3_, expectedEndLumiProd_, tag, "endLumi", lumi);

    edm::InputTag tagnew("thingWithMergeProducer", "endLumi");
    checkExpectedLumiProducts(index7_, expectedEndLumiNew_, tagnew, "endLumi", lumi);
  }

  void TestMergeResults::respondToOpenInputFile(FileBlock const& fb) {

    index0_ += 3;
    index1_ += 3;
    index2_ += 3;
    index3_ += 3;
    index4_ += 3;
    index5_ += 3;
    index6_ += 3;
    index7_ += 3;


    if (verbose_) edm::LogInfo("TestMergeResults") << "respondToOpenInputFile";

    if (!expectedInputFileNames_.empty()) {
      if (expectedInputFileNames_.size() <= static_cast<unsigned>(nRespondToOpenInputFile_) ||
          expectedInputFileNames_[nRespondToOpenInputFile_] != fb.fileName()) {
        std::cerr << "Error while testing merging of run/lumi products in TestMergeResults.cc\n"
                  << "Unexpected input filename, expected name = " << expectedInputFileNames_[nRespondToOpenInputFile_]
                  << "    actual name = " << fb.fileName() << std::endl; 
        abort();
      }
    }
    ++nRespondToOpenInputFile_;
  }

  void TestMergeResults::respondToCloseInputFile(FileBlock const& fb) {
    if (verbose_) edm::LogInfo("TestMergeResults") << "respondToCloseInputFile";
    ++nRespondToCloseInputFile_;
  }

  void TestMergeResults::respondToOpenOutputFiles(FileBlock const& fb) {
    if (verbose_) edm::LogInfo("TestMergeResults") << "respondToOpenOutputFiles";
    ++nRespondToOpenOutputFiles_;
  }

  void TestMergeResults::respondToCloseOutputFiles(FileBlock const& fb) {
    if (verbose_) edm::LogInfo("TestMergeResults") << "respondToCloseOutputFiles";
    ++nRespondToCloseOutputFiles_;
  }

  void TestMergeResults::endJob() {
    if (verbose_) edm::LogInfo("TestMergeResults") << "endJob";


    if (expectedRespondToOpenInputFile_ > -1 && nRespondToOpenInputFile_ != expectedRespondToOpenInputFile_) {
      std::cerr << "Error while testing merging of run/lumi products in TestMergeResults.cc\n"
              << "Unexpected number of calls to the function respondToOpenInputFile" << std::endl; 
      abort();
    }

    if (expectedRespondToCloseInputFile_ > -1 && nRespondToCloseInputFile_ != expectedRespondToCloseInputFile_) {
      std::cerr << "Error while testing merging of run/lumi products in TestMergeResults.cc\n"
              << "Unexpected number of calls to the function respondToCloseInputFile" << std::endl; 
      abort();
    }

    if (expectedRespondToOpenOutputFiles_ > -1 && nRespondToOpenOutputFiles_ != expectedRespondToOpenOutputFiles_) {
      std::cerr << "Error while testing merging of run/lumi products in TestMergeResults.cc\n"
              << "Unexpected number of calls to the function respondToOpenOutputFiles" << std::endl; 
      abort();
    }

    if (expectedRespondToCloseOutputFiles_ > -1 && nRespondToCloseOutputFiles_ != expectedRespondToCloseOutputFiles_) {
      std::cerr << "Error while testing merging of run/lumi products in TestMergeResults.cc\n"
              << "Unexpected number of calls to the function respondToCloseOutputFiles" << std::endl; 
      abort();
    }
  }

  void
  TestMergeResults::checkExpectedRunProducts(unsigned int index,
                                             std::vector<int> const& expectedValues,
                                             InputTag const& tag,
                                             const char* functionName,
                                             Run const& run) {

    if ((index + 2) < expectedValues.size()) {

      int expected = expectedValues[index];
      if (expected != 0) {
        run.getByLabel(tag, h_thing);
        if (h_thing->a != expected) 
          abortWithMessage(functionName, "Thing", tag, expected, h_thing->a);
      }

      expected = expectedValues[index + 1];
      if (expected != 0) {
        run.getByLabel(tag, h_thingWithMerge);
        if (h_thingWithMerge->a != expected) 
          abortWithMessage(functionName, "ThingWithMerge", tag, expected, h_thingWithMerge->a);
      }

      expected = expectedValues[index + 2];
      if (expected != 0) {
        run.getByLabel(tag, h_thingWithIsEqual);
        if (h_thingWithIsEqual->a != expected) 
          abortWithMessage(functionName, "ThingWithIsEqual", tag, expected, h_thingWithIsEqual->a);
      }
    }
  }

  void
  TestMergeResults::checkExpectedLumiProducts(unsigned int index,
                                              std::vector<int> const& expectedValues,
                                              InputTag const& tag,
                                              const char* functionName,
                                              LuminosityBlock const& lumi) {

    if ((index + 2) < expectedValues.size()) {

      int expected = expectedValues[index];
      if (expected != 0) {
        lumi.getByLabel(tag, h_thing);
        if (h_thing->a != expected) 
          abortWithMessage(functionName, "Thing", tag, expected, h_thing->a);
      }

      expected = expectedValues[index + 1];
      if (expected != 0) {
        lumi.getByLabel(tag, h_thingWithMerge);
        if (h_thingWithMerge->a != expected) 
          abortWithMessage(functionName, "ThingWithMerge", tag, expected, h_thingWithMerge->a);
      }

      expected = expectedValues[index + 2];
      if (expected != 0) {
        lumi.getByLabel(tag, h_thingWithIsEqual);
        if (h_thingWithIsEqual->a != expected) 
          abortWithMessage(functionName, "ThingWithIsEqual", tag, expected, h_thingWithIsEqual->a);
      }
    }
  }

  void TestMergeResults::abortWithMessage(const char* whichFunction, const char* type, edm::InputTag const& tag,
                                          int expectedValue, int actualValue) const {
    std::cerr << "Error while testing merging of run/lumi products in TestMergeResults.cc\n"
              << "In function " << whichFunction << " looking for product of type " << type << "\n"
              << tag << "\n"
              << "Expected value = " << expectedValue << " actual value = " << actualValue << std::endl; 
    abort();
  }
}

using edmtest::TestMergeResults;

DEFINE_FWK_MODULE(TestMergeResults);
