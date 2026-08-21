
//
// Reads some simple test objects in the event, run, and lumi
// principals.  Then checks to see if the values in these
// objects match what we expect.  Intended to be used to
// test the values in a file that has merged run and lumi
// products.
//
// Original Author: David Dagenhart, Fermilab, February 2008

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingWithIsEqual.h"
#include "DataFormats/TestObjects/interface/ThingWithMerge.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace edm {
  class EventSetup;
}

// Captures the call site line number and forwards it (as the first argument) to reportProblem().
#define REPORT_PROBLEM(...) reportProblem(__LINE__, __VA_ARGS__)

namespace edmtest {

  class TestMergeResultsOutputModule : public edm::one::OutputModule<> {
  public:
    explicit TestMergeResultsOutputModule(edm::ParameterSet const&);

    static void fillDescriptions(edm::ConfigurationDescriptions&);

    void write(edm::EventForOutput const& e) final;
    void writeRun(edm::RunForOutput const&) final;
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) final;
    void endJob() override;

  private:
    void checkExpectedLumiProducts(unsigned int index,
                                   std::vector<int> const& expectedValues,
                                   edm::EDGetTokenT<edmtest::Thing> const& thingToken,
                                   edm::EDGetTokenT<edmtest::ThingWithMerge> const& mergeToken,
                                   edm::EDGetTokenT<edmtest::ThingWithIsEqual> const& isEqualToken,
                                   edm::InputTag const& tag,
                                   char const* functionName,
                                   edm::LuminosityBlockForOutput const& lumi,
                                   std::vector<int> const& expectedValueImproperlyMerged);

    void checkExpectedRunProducts(unsigned int index,
                                  std::vector<int> const& expectedValues,
                                  edm::EDGetTokenT<edmtest::Thing> const& thingToken,
                                  edm::EDGetTokenT<edmtest::ThingWithMerge> const& mergeToken,
                                  edm::EDGetTokenT<edmtest::ThingWithIsEqual> const& isEqualToken,
                                  edm::InputTag const& tag,
                                  char const* functionName,
                                  edm::RunForOutput const& run,
                                  std::vector<int> const& expectedValueImproperlyMerged);

    void reportProblem(int line,
                       char const* whichFunction,
                       char const* type,
                       edm::InputTag const& tag,
                       int expectedValue,
                       int actualValue,
                       bool unexpectedImproperlyMergedValue = false);

    void reportProblem(int line, std::string const& message);

    std::vector<int> const expectedBeginRunProd_;
    std::vector<int> const expectedEndRunProd_;
    std::vector<int> const expectedBeginLumiProd_;
    std::vector<int> const expectedEndLumiProd_;

    std::vector<int> const expectedBeginRunNew_;
    std::vector<int> const expectedEndRunNew_;
    std::vector<int> const expectedBeginLumiNew_;
    std::vector<int> const expectedEndLumiNew_;

    std::vector<int> const expectedEndRunProdImproperlyMerged_;
    std::vector<int> const expectedEndLumiProdImproperlyMerged_;

    std::vector<std::string> const expectedParents_;

    std::vector<std::string> const expectedProcessHistoryInRuns_;

    std::vector<int> const expectedDroppedEvent_;
    std::vector<int> const expectedDroppedEvent1_;
    std::vector<int> const expectedDroppedEvent1NEvents_;

    bool const verbose_;
    bool const testAlias_;

    unsigned int indexRun_ = 0;
    unsigned int indexLumi_ = 0;
    unsigned int parentIndex_ = 0;
    unsigned int droppedIndex1_ = 0;
    int droppedIndex1EventCount_ = 0;
    unsigned int processHistoryIndex_ = 0;
    unsigned int problemCount_ = 0;

    edm::Handle<edmtest::Thing> h_thing;
    edm::Handle<edmtest::ThingWithMerge> h_thingWithMerge;
    edm::Handle<edmtest::ThingWithIsEqual> h_thingWithIsEqual;

    // Event branch tokens
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> eventDroppedIsEqualToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> eventDroppedMergeToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> eventDropped1IsEqualToken_;
    std::unordered_map<std::string, edm::EDGetTokenT<edmtest::Thing>> parentTokenMap_;
    edm::EDGetTokenT<edmtest::Thing> eventThingToken_;
    edm::EDGetTokenT<edmtest::Thing> eventAliasToken_;
    edm::EDGetTokenT<edmtest::Thing> eventAliasPRODToken_;

    // Run branch tokens
    edm::EDGetTokenT<edmtest::Thing> runBeginProdThingToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> runBeginProdMergeToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> runBeginProdIsEqualToken_;
    edm::EDGetTokenT<edmtest::Thing> runBeginNewThingToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> runBeginNewMergeToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> runBeginNewIsEqualToken_;
    edm::EDGetTokenT<edmtest::Thing> runEndProdThingToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> runEndProdMergeToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> runEndProdIsEqualToken_;
    edm::EDGetTokenT<edmtest::Thing> runEndNewThingToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> runEndNewMergeToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> runEndNewIsEqualToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> runDroppedMergeToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> runDroppedIsEqualToken_;
    edm::EDGetTokenT<edmtest::Thing> runAliasToken_;
    edm::EDGetTokenT<edmtest::Thing> runAliasPRODToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> runBeginDroppedIsEqualToken_;

    // LuminosityBlock branch tokens
    edm::EDGetTokenT<edmtest::Thing> lumiBeginProdThingToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> lumiBeginProdMergeToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> lumiBeginProdIsEqualToken_;
    edm::EDGetTokenT<edmtest::Thing> lumiBeginNewThingToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> lumiBeginNewMergeToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> lumiBeginNewIsEqualToken_;
    edm::EDGetTokenT<edmtest::Thing> lumiEndProdThingToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> lumiEndProdMergeToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> lumiEndProdIsEqualToken_;
    edm::EDGetTokenT<edmtest::Thing> lumiEndNewThingToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> lumiEndNewMergeToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> lumiEndNewIsEqualToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> lumiDroppedBeginIsEqualToken_;
    edm::EDGetTokenT<edmtest::ThingWithIsEqual> lumiDroppedEndIsEqualToken_;
    edm::EDGetTokenT<edmtest::ThingWithMerge> lumiDroppedEndMergeToken_;
    edm::EDGetTokenT<edmtest::Thing> lumiAliasToken_;
    edm::EDGetTokenT<edmtest::Thing> lumiAliasPRODToken_;
  };

  // -----------------------------------------------------------------

  TestMergeResultsOutputModule::TestMergeResultsOutputModule(edm::ParameterSet const& ps)
      : edm::one::OutputModuleBase(ps),
        edm::one::OutputModule<>(ps),
        expectedBeginRunProd_(ps.getUntrackedParameter<std::vector<int>>("expectedBeginRunProd")),
        expectedEndRunProd_(ps.getUntrackedParameter<std::vector<int>>("expectedEndRunProd")),
        expectedBeginLumiProd_(ps.getUntrackedParameter<std::vector<int>>("expectedBeginLumiProd")),
        expectedEndLumiProd_(ps.getUntrackedParameter<std::vector<int>>("expectedEndLumiProd")),

        expectedBeginRunNew_(ps.getUntrackedParameter<std::vector<int>>("expectedBeginRunNew")),
        expectedEndRunNew_(ps.getUntrackedParameter<std::vector<int>>("expectedEndRunNew")),
        expectedBeginLumiNew_(ps.getUntrackedParameter<std::vector<int>>("expectedBeginLumiNew")),
        expectedEndLumiNew_(ps.getUntrackedParameter<std::vector<int>>("expectedEndLumiNew")),

        expectedEndRunProdImproperlyMerged_(
            ps.getUntrackedParameter<std::vector<int>>("expectedEndRunProdImproperlyMerged")),
        expectedEndLumiProdImproperlyMerged_(
            ps.getUntrackedParameter<std::vector<int>>("expectedEndLumiProdImproperlyMerged")),

        expectedParents_(ps.getUntrackedParameter<std::vector<std::string>>("expectedParents")),
        expectedProcessHistoryInRuns_(
            ps.getUntrackedParameter<std::vector<std::string>>("expectedProcessHistoryInRuns")),

        expectedDroppedEvent_(ps.getUntrackedParameter<std::vector<int>>("expectedDroppedEvent")),
        expectedDroppedEvent1_(ps.getUntrackedParameter<std::vector<int>>("expectedDroppedEvent1")),
        expectedDroppedEvent1NEvents_(ps.getUntrackedParameter<std::vector<int>>("expectedDroppedEvent1NEvents")),

        verbose_(ps.getUntrackedParameter<bool>("verbose")),
        testAlias_(ps.getUntrackedParameter<bool>("testAlias")) {
    auto ap_thing = std::make_unique<edmtest::Thing>();
    edm::Wrapper<edmtest::Thing> w_thing(std::move(ap_thing));
    assert(!w_thing.isMergeable());
    assert(!w_thing.hasIsProductEqual());
    assert(!w_thing.hasSwap());

    auto ap_thingwithmerge = std::make_unique<edmtest::ThingWithMerge>();
    edm::Wrapper<edmtest::ThingWithMerge> w_thingWithMerge(std::move(ap_thingwithmerge));
    assert(w_thingWithMerge.isMergeable());
    assert(!w_thingWithMerge.hasIsProductEqual());
    assert(w_thingWithMerge.hasSwap());

    auto ap_thingwithisequal = std::make_unique<edmtest::ThingWithIsEqual>();
    edm::Wrapper<edmtest::ThingWithIsEqual> w_thingWithIsEqual(std::move(ap_thingwithisequal));
    assert(!w_thingWithIsEqual.isMergeable());
    assert(w_thingWithIsEqual.hasIsProductEqual());
    assert(!w_thingWithIsEqual.hasSwap());

    if (!expectedDroppedEvent_.empty()) {
      eventDroppedIsEqualToken_ =
          consumes<edmtest::ThingWithIsEqual>(edm::InputTag{"makeThingToBeDropped", "event", "PROD"});
      eventDroppedMergeToken_ =
          consumes<edmtest::ThingWithMerge>(edm::InputTag{"makeThingToBeDropped", "event", "PROD"});

      runBeginDroppedIsEqualToken_ =
          consumes<edmtest::ThingWithIsEqual, edm::InRun>(edm::InputTag{"makeThingToBeDropped", "beginRun", "PROD"});
    }
    for (auto const& parent : expectedParents_) {
      parentTokenMap_.try_emplace(parent, consumes<edmtest::Thing>(edm::InputTag{parent, "event", "PROD"}));
    }
    if (expectedDroppedEvent1_.size() > droppedIndex1_) {
      assert(expectedDroppedEvent1_.size() == expectedDroppedEvent1NEvents_.size());
      eventDropped1IsEqualToken_ =
          consumes<edmtest::ThingWithIsEqual>(edm::InputTag{"makeThingToBeDropped1", "event", "PROD"});
    }
    eventThingToken_ = consumes<edmtest::Thing>(edm::InputTag{"thingWithMergeProducer", "event", "PROD"});

    if (testAlias_) {
      eventAliasToken_ = consumes<edmtest::Thing>(edm::InputTag{"aliasForThingToBeDropped2", "instance2"});
      eventAliasPRODToken_ = consumes<edmtest::Thing>(edm::InputTag{"aliasForThingToBeDropped2", "instance2", "PROD"});
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "beginRun", "PROD");
      runBeginProdThingToken_ = consumes<edmtest::Thing, edm::InRun>(tag);
      runBeginProdMergeToken_ = consumes<edmtest::ThingWithMerge, edm::InRun>(tag);
      runBeginProdIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InRun>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "beginRun");
      runBeginNewThingToken_ = consumes<edmtest::Thing, edm::InRun>(tag);
      runBeginNewMergeToken_ = consumes<edmtest::ThingWithMerge, edm::InRun>(tag);
      runBeginNewIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InRun>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "endRun", "PROD");
      runEndProdThingToken_ = consumes<edmtest::Thing, edm::InRun>(tag);
      runEndProdMergeToken_ = consumes<edmtest::ThingWithMerge, edm::InRun>(tag);
      runEndProdIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InRun>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "endRun");
      runEndNewThingToken_ = consumes<edmtest::Thing, edm::InRun>(tag);
      runEndNewMergeToken_ = consumes<edmtest::ThingWithMerge, edm::InRun>(tag);
      runEndNewIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InRun>(tag);
    }

    if (expectedDroppedEvent_.size() > 2) {
      edm::InputTag tag("makeThingToBeDropped", "endRun", "PROD");
      runDroppedMergeToken_ = consumes<edmtest::ThingWithMerge, edm::InRun>(tag);
      runDroppedIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InRun>(tag);
    }

    if (testAlias_) {
      runAliasToken_ = consumes<edmtest::Thing, edm::InRun>(edm::InputTag{"aliasForThingToBeDropped2", "endRun2"});
      edm::InputTag tag("aliasForThingToBeDropped2", "endRun2", "PROD");
      runAliasPRODToken_ = consumes<edmtest::Thing, edm::InRun>(tag);
    }

    if (expectedDroppedEvent_.size() > 3) {
      edm::InputTag tag("makeThingToBeDropped", "beginLumi", "PROD");
      lumiDroppedBeginIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InLumi>(tag);
    }
    {
      edm::InputTag tag("thingWithMergeProducer", "endLumi", "PROD");
      lumiEndProdThingToken_ = consumes<edmtest::Thing, edm::InLumi>(tag);
      lumiEndProdMergeToken_ = consumes<edmtest::ThingWithMerge, edm::InLumi>(tag);
      lumiEndProdIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InLumi>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "endLumi");
      lumiEndNewThingToken_ = consumes<edmtest::Thing, edm::InLumi>(tag);
      lumiEndNewMergeToken_ = consumes<edmtest::ThingWithMerge, edm::InLumi>(tag);
      lumiEndNewIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InLumi>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "beginLumi", "PROD");
      lumiBeginProdThingToken_ = consumes<edmtest::Thing, edm::InLumi>(tag);
      lumiBeginProdMergeToken_ = consumes<edmtest::ThingWithMerge, edm::InLumi>(tag);
      lumiBeginProdIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InLumi>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "beginLumi");
      lumiBeginNewThingToken_ = consumes<edmtest::Thing, edm::InLumi>(tag);
      lumiBeginNewMergeToken_ = consumes<edmtest::ThingWithMerge, edm::InLumi>(tag);
      lumiBeginNewIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InLumi>(tag);
    }

    if (expectedDroppedEvent_.size() > 4) {
      edm::InputTag tag("makeThingToBeDropped", "endLumi", "PROD");
      lumiDroppedEndIsEqualToken_ = consumes<edmtest::ThingWithIsEqual, edm::InLumi>(tag);
      lumiDroppedEndMergeToken_ = consumes<edmtest::ThingWithMerge, edm::InLumi>(tag);
    }

    if (testAlias_) {
      lumiAliasToken_ = consumes<edmtest::Thing, edm::InLumi>(edm::InputTag{"aliasForThingToBeDropped2", "endLumi2"});
      edm::InputTag tag("aliasForThingToBeDropped2", "endLumi2", "PROD");
      lumiAliasPRODToken_ = consumes<edmtest::Thing, edm::InLumi>(tag);
    }
  }

  // -----------------------------------------------------------------

  void TestMergeResultsOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<std::vector<int>>("expectedBeginRunProd", {})
        ->setComment(
            "Check the expected values of Thing, ThingWithMerge, ThingWithIsEqual from process PROD at beginRun.");
    desc.addUntracked<std::vector<int>>("expectedEndRunProd", {})
        ->setComment(
            "Check the expected values of Thing, ThingWithMerge, ThingWithIsEqual from process PROD at nendRun.");
    desc.addUntracked<std::vector<int>>("expectedBeginLumiProd", {})
        ->setComment(
            "Check the expected values of Thing, ThingWithMerge, ThingWithIsEqual from process PROD at "
            "beginLuminosityBlock.");
    desc.addUntracked<std::vector<int>>("expectedEndLumiProd", {})
        ->setComment(
            "Check the expected values of Thing, ThingWithMerge, ThingWithIsEqual from process PROD at "
            "endLuminosityBlock.");

    desc.addUntracked<std::vector<int>>("expectedBeginRunNew", {})
        ->setComment(
            "Check the expected values of Thing, ThingWithMerge, ThingWithIsEqual from the latest process at "
            "beginRun.");
    desc.addUntracked<std::vector<int>>("expectedEndRunNew", {})
        ->setComment(
            "Check the expected values of Thing, ThingWithMerge, ThingWithIsEqual from the latest process at endRun.");
    desc.addUntracked<std::vector<int>>("expectedBeginLumiNew", {})
        ->setComment(
            "Check the expected values of Thing, ThingWithMerge, ThingWithIsEqual from the latest process at "
            "beginLuminosityBlock.");
    desc.addUntracked<std::vector<int>>("expectedEndLumiNew", {})
        ->setComment(
            "Check the expected values of Thing, ThingWithMerge, ThingWithIsEqual from the latest process at "
            "endLuminosityBlock.");

    desc.addUntracked<std::vector<int>>("expectedEndRunProdImproperlyMerged", {});
    desc.addUntracked<std::vector<int>>("expectedEndLumiProdImproperlyMerged", {});
    desc.addUntracked<std::vector<std::string>>("expectedParents", {});
    desc.addUntracked<std::vector<std::string>>("expectedProcessHistoryInRuns", {});
    desc.addUntracked<std::vector<int>>("expectedDroppedEvent", {});
    desc.addUntracked<std::vector<int>>("expectedDroppedEvent1", {});
    desc.addUntracked<std::vector<int>>("expectedDroppedEvent1NEvents", {});

    desc.addUntracked<bool>("testAlias", false);
    desc.addUntracked<bool>("verbose", false);

    desc.setComment(
        "The expected{Begin,End}(Run,Lumi}{Prod,New} parameters follow the same pattern. The expected values come in "
        "sets of three: value expected in Thing, ThingWithMerge, and ThingWithIsEqual. Each set of 3 is tested at the "
        "specific transition, and then the next set of 3 is tested at the next transition, and so on. When the "
        "sequence of parameter values is exhausted, the checking is stopped. The values if 0 are just placedholders, "
        "i.e. if the value is a 0, the check is not made.");

    descriptions.addDefault(desc);
  }

  // -----------------------------------------------------------------
  void TestMergeResultsOutputModule::write(edm::EventForOutput const& e) {
    assert(e.processHistory().id() == e.processHistoryID());

    if (verbose_)
      edm::LogInfo("TestMergeResults") << "analyze";

    if (!expectedDroppedEvent_.empty()) {
      h_thingWithIsEqual = e.getHandle(eventDroppedIsEqualToken_);
      assert(h_thingWithIsEqual->a == expectedDroppedEvent_[0]);

      h_thingWithMerge = e.getHandle(eventDroppedMergeToken_);
      assert(h_thingWithMerge.isValid());
    }

    // This one is used to test the merging step when a specific product
    // has been dropped or not created in some of the input files.
    if (expectedDroppedEvent1_.size() > droppedIndex1_) {
      ++droppedIndex1EventCount_;
      if (droppedIndex1EventCount_ > expectedDroppedEvent1NEvents_[droppedIndex1_]) {
        ++droppedIndex1_;
        droppedIndex1EventCount_ = 1;
      }
      assert(droppedIndex1_ < expectedDroppedEvent1_.size());

      h_thingWithIsEqual = e.getHandle(eventDropped1IsEqualToken_);
      if (expectedDroppedEvent1_[droppedIndex1_] == -1) {
        assert(!h_thingWithIsEqual.isValid());
      } else {
        assert(h_thingWithIsEqual.isValid());
        assert(h_thingWithIsEqual->a == expectedDroppedEvent1_[droppedIndex1_]);
      }
    }

    // I'm not sure this test belongs in this module.  Originally it tested
    // merging of parentage for run and lumi products, but at some point the
    // parentage for run/lumi products stopped being written at all so there was
    // nothing to test.  This was the only real test of the provenance
    // parentage, so I just converted to a test of the parentage of products
    // in the Event rather than deleting it or writing a complete new test ...
    // It is actually convenient here, so maybe it is OK even if the module name
    // has nothing to do with this test.
    if (parentIndex_ < expectedParents_.size()) {
      h_thing = e.getHandle(eventThingToken_);
      std::string expectedParent = expectedParents_[parentIndex_];
      edm::BranchID actualParentBranchID = h_thing.provenance()->productProvenance()->parentage().parents()[0];

      // There ought to be a get that uses the BranchID as an argument, but
      // there is not at the moment so we get the Provenance first and use that
      // find the actual parent
      edm::Provenance prov = e.getProvenance(actualParentBranchID);
      assert(expectedParent == prov.moduleLabel());
      auto tokenIt = parentTokenMap_.find(prov.moduleLabel());
      assert(tokenIt != parentTokenMap_.end());
      h_thing = e.getHandle(tokenIt->second);
      assert(h_thing->a == 11);
      ++parentIndex_;
    }

    if (testAlias_) {
      h_thing = e.getHandle(eventAliasToken_);
      assert(h_thing->a == 11);
      h_thing = e.getHandle(eventAliasPRODToken_);
      assert(h_thing->a == 11);

      edm::BranchID const& originalBranchID = h_thing.provenance()->productDescription().originalBranchID();
      //this will throw if the original provenance is not available
      e.getProvenance(originalBranchID);
    }
  }

  void TestMergeResultsOutputModule::writeRun(edm::RunForOutput const& run) {
    assert(run.processHistory().id() == run.processHistoryID());

    edm::ProcessHistory const& ph = run.processHistory();
    for (edm::ProcessHistory::const_iterator iter = ph.begin(), iEnd = ph.end(); iter != iEnd; ++iter) {
      if (processHistoryIndex_ < expectedProcessHistoryInRuns_.size()) {
        assert(expectedProcessHistoryInRuns_[processHistoryIndex_] == iter->processName());
        ++processHistoryIndex_;
      }
    }

    if (verbose_)
      edm::LogInfo("TestMergeResults") << "endRun";

    std::vector<int> emptyDummy;

    edm::InputTag tag("thingWithMergeProducer", "endRun", "PROD");
    checkExpectedRunProducts(indexRun_,
                             expectedEndRunProd_,
                             runEndProdThingToken_,
                             runEndProdMergeToken_,
                             runEndProdIsEqualToken_,
                             tag,
                             "endRun",
                             run,
                             expectedEndRunProdImproperlyMerged_);

    edm::InputTag tagnew("thingWithMergeProducer", "endRun");
    checkExpectedRunProducts(indexRun_,
                             expectedEndRunNew_,
                             runEndNewThingToken_,
                             runEndNewMergeToken_,
                             runEndNewIsEqualToken_,
                             tagnew,
                             "endRun",
                             run,
                             emptyDummy);

    edm::InputTag tagb("thingWithMergeProducer", "beginRun", "PROD");
    checkExpectedRunProducts(indexRun_,
                             expectedBeginRunProd_,
                             runBeginProdThingToken_,
                             runBeginProdMergeToken_,
                             runBeginProdIsEqualToken_,
                             tagb,
                             "endRun",
                             run,
                             emptyDummy);

    edm::InputTag tagbnew("thingWithMergeProducer", "beginRun");
    checkExpectedRunProducts(indexRun_,
                             expectedBeginRunNew_,
                             runBeginNewThingToken_,
                             runBeginNewMergeToken_,
                             runBeginNewIsEqualToken_,
                             tagbnew,
                             "endRun",
                             run,
                             emptyDummy);

    if (expectedDroppedEvent_.size() > 2) {
      h_thingWithIsEqual = run.getHandle(runDroppedIsEqualToken_);
      assert(h_thingWithIsEqual->a == expectedDroppedEvent_[2]);

      h_thingWithMerge = run.getHandle(runDroppedMergeToken_);
      assert(!h_thingWithMerge.isValid());
    }

    if (testAlias_) {
      h_thing = run.getHandle(runAliasToken_);
      assert(h_thing->a == 100001);
      h_thing = run.getHandle(runAliasPRODToken_);
      assert(h_thing->a == 100001);

      edm::BranchID const& originalBranchID = h_thing.provenance()->productDescription().originalBranchID();
      run.getProvenance(originalBranchID);
    }

    indexRun_ += 3;
  }

  void TestMergeResultsOutputModule::writeLuminosityBlock(edm::LuminosityBlockForOutput const& lumi) {
    assert(lumi.processHistory().id() == lumi.processHistoryID());

    if (verbose_)
      edm::LogInfo("TestMergeResults") << "endLuminosityBlock";

    std::vector<int> emptyDummy;

    edm::InputTag tag("thingWithMergeProducer", "endLumi", "PROD");
    checkExpectedLumiProducts(indexLumi_,
                              expectedEndLumiProd_,
                              lumiEndProdThingToken_,
                              lumiEndProdMergeToken_,
                              lumiEndProdIsEqualToken_,
                              tag,
                              "endLumi",
                              lumi,
                              expectedEndLumiProdImproperlyMerged_);

    edm::InputTag tagnew("thingWithMergeProducer", "endLumi");
    checkExpectedLumiProducts(indexLumi_,
                              expectedEndLumiNew_,
                              lumiEndNewThingToken_,
                              lumiEndNewMergeToken_,
                              lumiEndNewIsEqualToken_,
                              tagnew,
                              "endLumi",
                              lumi,
                              emptyDummy);

    edm::InputTag tagb("thingWithMergeProducer", "beginLumi", "PROD");
    checkExpectedLumiProducts(indexLumi_,
                              expectedBeginLumiProd_,
                              lumiBeginProdThingToken_,
                              lumiBeginProdMergeToken_,
                              lumiBeginProdIsEqualToken_,
                              tagb,
                              "endLumi",
                              lumi,
                              emptyDummy);

    edm::InputTag tagbnew("thingWithMergeProducer", "beginLumi");
    checkExpectedLumiProducts(indexLumi_,
                              expectedBeginLumiNew_,
                              lumiBeginNewThingToken_,
                              lumiBeginNewMergeToken_,
                              lumiBeginNewIsEqualToken_,
                              tagbnew,
                              "endLumi",
                              lumi,
                              emptyDummy);

    if (expectedDroppedEvent_.size() > 4) {
      h_thingWithIsEqual = lumi.getHandle(lumiDroppedEndIsEqualToken_);
      assert(h_thingWithIsEqual->a == expectedDroppedEvent_[4]);

      h_thingWithMerge = lumi.getHandle(lumiDroppedEndMergeToken_);
      assert(!h_thingWithMerge.isValid());
    }

    if (testAlias_) {
      h_thing = lumi.getHandle(lumiAliasToken_);
      assert(h_thing->a == 1001);
      h_thing = lumi.getHandle(lumiAliasPRODToken_);
      assert(h_thing->a == 1001);

      edm::BranchID const& originalBranchID = h_thing.provenance()->productDescription().originalBranchID();
      lumi.getProvenance(originalBranchID);
    }
    indexLumi_ += 3;
  }

  void TestMergeResultsOutputModule::endJob() {
    if (verbose_)
      edm::LogInfo("TestMergeResults") << "endJob";
    if (problemCount_ > 0) {
      throw cms::Exception("TestMergeResults") << problemCount_ << " problem(s) found, see messages above.";
    }
  }

  void TestMergeResultsOutputModule::checkExpectedRunProducts(
      unsigned int index,
      std::vector<int> const& expectedValues,
      edm::EDGetTokenT<edmtest::Thing> const& thingToken,
      edm::EDGetTokenT<edmtest::ThingWithMerge> const& mergeToken,
      edm::EDGetTokenT<edmtest::ThingWithIsEqual> const& isEqualToken,
      edm::InputTag const& tag,
      char const* functionName,
      edm::RunForOutput const& run,
      std::vector<int> const& expectedValueImproperlyMerged) {
    if ((index + 2) < expectedValues.size()) {
      int expected = expectedValues[index];
      if (expected != 0) {
        h_thing = run.getHandle(thingToken);
        if (h_thing->a != expected) {
          REPORT_PROBLEM(functionName, "Thing", tag, expected, h_thing->a);
        }
        if (index < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index] != 0) != h_thing.provenance()->knownImproperlyMerged()) {
            REPORT_PROBLEM(functionName, "Thing", tag, 0, 0, true);
          }
        }
      }

      expected = expectedValues[index + 1];
      if (expected != 0) {
        h_thingWithMerge = run.getHandle(mergeToken);
        if (h_thingWithMerge->a != expected) {
          REPORT_PROBLEM(functionName, "ThingWithMerge", tag, expected, h_thingWithMerge->a);
        }
        if (index + 1 < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index + 1] != 0) !=
              h_thingWithMerge.provenance()->knownImproperlyMerged()) {
            REPORT_PROBLEM(functionName, "ThingWithMerge", tag, 0, 0, true);
          }
        }
        if (!h_thingWithMerge.provenance()->productDescription().isMergeable()) {
          REPORT_PROBLEM(
              "TestMergeResults::checkExpectedRunProducts isMergeable from ProductDescription returns "
              "unexpected value for ThingWithMerge type.");
        }
      }

      expected = expectedValues[index + 2];
      if (expected != 0) {
        h_thingWithIsEqual = run.getHandle(isEqualToken);
        if (h_thingWithIsEqual->a != expected) {
          REPORT_PROBLEM(functionName, "ThingWithIsEqual", tag, expected, h_thingWithIsEqual->a);
        }
        if (index + 2 < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index + 2] != 0) !=
              h_thingWithIsEqual.provenance()->knownImproperlyMerged()) {
            REPORT_PROBLEM(functionName, "ThingWithIsEqual", tag, 0, 0, true);
          }
        }
        if (h_thingWithIsEqual.provenance()->productDescription().isMergeable()) {
          REPORT_PROBLEM(
              "TestMergeResults::checkExpectedRunProducts isMergeable from ProductDescription returns "
              "unexpected value for ThingWithIsEqual type.");
        }
      }
    }
  }

  void TestMergeResultsOutputModule::checkExpectedLumiProducts(
      unsigned int index,
      std::vector<int> const& expectedValues,
      edm::EDGetTokenT<edmtest::Thing> const& thingToken,
      edm::EDGetTokenT<edmtest::ThingWithMerge> const& mergeToken,
      edm::EDGetTokenT<edmtest::ThingWithIsEqual> const& isEqualToken,
      edm::InputTag const& tag,
      char const* functionName,
      edm::LuminosityBlockForOutput const& lumi,
      std::vector<int> const& expectedValueImproperlyMerged) {
    if ((index + 2) < expectedValues.size()) {
      int expected = expectedValues[index];
      if (expected != 0) {
        h_thing = lumi.getHandle(thingToken);
        if (h_thing->a != expected) {
          REPORT_PROBLEM(functionName, "Thing", tag, expected, h_thing->a);
        }
        if (index < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index] != 0) != h_thing.provenance()->knownImproperlyMerged()) {
            REPORT_PROBLEM(functionName, "Thing", tag, 0, 0, true);
          }
        }
      }

      expected = expectedValues[index + 1];
      if (expected != 0) {
        h_thingWithMerge = lumi.getHandle(mergeToken);
        if (h_thingWithMerge->a != expected) {
          REPORT_PROBLEM(functionName, "ThingWithMerge", tag, expected, h_thingWithMerge->a);
        }
        if (index + 1 < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index + 1] != 0) !=
              h_thingWithMerge.provenance()->knownImproperlyMerged()) {
            REPORT_PROBLEM(functionName, "ThingWithMerge", tag, 0, 0, true);
          }
        }
        if (!h_thingWithMerge.provenance()->productDescription().isMergeable()) {
          REPORT_PROBLEM(
              "TestMergeResults::checkExpectedLumiProducts isMergeable from ProductDescription returns "
              "unexpected value for ThingWithMerge type.");
        }
      }

      expected = expectedValues[index + 2];
      if (expected != 0) {
        h_thingWithIsEqual = lumi.getHandle(isEqualToken);
        if (h_thingWithIsEqual->a != expected) {
          REPORT_PROBLEM(functionName, "ThingWithIsEqual", tag, expected, h_thingWithIsEqual->a);
        }
        if (index + 2 < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index + 2] != 0) !=
              h_thingWithIsEqual.provenance()->knownImproperlyMerged()) {
            REPORT_PROBLEM(functionName, "ThingWithIsEqual", tag, 0, 0, true);
          }
        }
        if (h_thingWithIsEqual.provenance()->productDescription().isMergeable()) {
          REPORT_PROBLEM(
              "TestMergeResults::checkExpectedLumiProducts isMergeable from ProductDescription returns "
              "unexpected value for ThingWithIsEqual type.");
        }
      }
    }
  }

  void TestMergeResultsOutputModule::reportProblem(int line,
                                                   char const* whichFunction,
                                                   char const* type,
                                                   edm::InputTag const& tag,
                                                   int expectedValue,
                                                   int actualValue,
                                                   bool unexpectedImproperlyMergedValue) {
    ++problemCount_;
    edm::LogError("TestMergeResults") << "Error while testing merging of run/lumi products in TestMergeResults.cc:"
                                      << line << "\n"
                                      << "In function " << whichFunction << " looking for product of type " << type
                                      << "\n"
                                      << tag;
    if (unexpectedImproperlyMergedValue) {
      edm::LogError("TestMergeResults") << "Unexpected value of knownImproperlyMerged from provenance";
    } else {
      edm::LogError("TestMergeResults") << "Expected value = " << expectedValue << " actual value = " << actualValue;
    }
  }

  void TestMergeResultsOutputModule::reportProblem(int line, std::string const& message) {
    ++problemCount_;
    edm::LogError("TestMergeResults") << "Error while testing merging of run/lumi products in TestMergeResults.cc:"
                                      << line << "\n"
                                      << message;
  }
}  // namespace edmtest

using edmtest::TestMergeResultsOutputModule;

DEFINE_FWK_MODULE(TestMergeResultsOutputModule);
