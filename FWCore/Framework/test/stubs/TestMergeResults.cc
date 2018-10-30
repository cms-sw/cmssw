
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
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingWithIsEqual.h"
#include "DataFormats/TestObjects/interface/ThingWithMerge.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace edm {
  class EventSetup;
}

namespace edmtest {

  class TestMergeResults : public edm::EDAnalyzer {
  public:

    explicit TestMergeResults(edm::ParameterSet const&);
    virtual ~TestMergeResults();

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
    virtual void beginRun(edm::Run const&, edm::EventSetup const&);
    virtual void endRun(edm::Run const&, edm::EventSetup const&);
    virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
    virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
    virtual void respondToOpenInputFile(edm::FileBlock const& fb);
    virtual void respondToCloseInputFile(edm::FileBlock const& fb);
    void endJob();

  private:

    void checkExpectedLumiProducts(unsigned int index,
                                   std::vector<int> const& expectedValues,
                                   edm::InputTag const& tag,
                                   char const* functionName,
                                   edm::LuminosityBlock const& lumi,
                                   std::vector<int> const& expectedValueImproperlyMerged);

    void checkExpectedRunProducts(unsigned int index,
                                  std::vector<int> const& expectedValues,
                                  edm::InputTag const& tag,
                                  char const* functionName,
                                  edm::Run const& run,
                                  std::vector<int> const& expectedValueImproperlyMerged);

    void abortWithMessage(char const* whichFunction, char const* type, edm::InputTag const& tag,
                          int expectedValue, int actualValue, bool unexpectedImproperlyMergedValue = false) const;

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

    std::vector<int> expectedEndRunProdImproperlyMerged_;
    std::vector<int> expectedEndLumiProdImproperlyMerged_;

    std::vector<std::string> expectedParents_;

    std::vector<std::string> expectedProcessHistoryInRuns_;

    std::vector<int> expectedDroppedEvent_;
    std::vector<int> expectedDroppedEvent1_;

    int nRespondToOpenInputFile_;
    int nRespondToCloseInputFile_;
    int expectedRespondToOpenInputFile_;
    int expectedRespondToCloseInputFile_;

    std::vector<std::string> expectedInputFileNames_;

    bool verbose_;

    unsigned int indexRun_;
    unsigned int indexLumi_;
    unsigned int parentIndex_;
    unsigned int droppedIndex1_;
    unsigned int processHistoryIndex_;

    edm::Handle<edmtest::Thing> h_thing;
    edm::Handle<edmtest::ThingWithMerge> h_thingWithMerge;
    edm::Handle<edmtest::ThingWithIsEqual> h_thingWithIsEqual;

    bool testAlias_;
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

    expectedEndRunProdImproperlyMerged_(ps.getUntrackedParameter<std::vector<int> >("expectedEndRunProdImproperlyMerged", default_)),
    expectedEndLumiProdImproperlyMerged_(ps.getUntrackedParameter<std::vector<int> >("expectedEndLumiProdImproperlyMerged", default_)),

    expectedParents_(ps.getUntrackedParameter<std::vector<std::string> >("expectedParents", defaultvstring_)),
    expectedProcessHistoryInRuns_(ps.getUntrackedParameter<std::vector<std::string> >("expectedProcessHistoryInRuns", defaultvstring_)),

    expectedDroppedEvent_(ps.getUntrackedParameter<std::vector<int> >("expectedDroppedEvent", default_)),
    expectedDroppedEvent1_(ps.getUntrackedParameter<std::vector<int> >("expectedDroppedEvent1", default_)),

    nRespondToOpenInputFile_(0),
    nRespondToCloseInputFile_(0),
    expectedRespondToOpenInputFile_(ps.getUntrackedParameter<int>("expectedRespondToOpenInputFile", -1)),
    expectedRespondToCloseInputFile_(ps.getUntrackedParameter<int>("expectedRespondToCloseInputFile", -1)),

    expectedInputFileNames_(ps.getUntrackedParameter<std::vector<std::string> >("expectedInputFileNames", defaultvstring_)),

    verbose_(ps.getUntrackedParameter<bool>("verbose", false)),

    indexRun_(0),
    indexLumi_(0),
    parentIndex_(0),
    droppedIndex1_(0),
    processHistoryIndex_(0),
    testAlias_(ps.getUntrackedParameter<bool>("testAlias", false)) {

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

    if(expectedDroppedEvent_.size() > 0) {
      consumes<edmtest::ThingWithIsEqual>(edm::InputTag{"makeThingToBeDropped", "event", "PROD"});
      consumes<edmtest::ThingWithMerge>(edm::InputTag{"makeThingToBeDropped", "event", "PROD"});

      consumes<edmtest::ThingWithIsEqual,edm::InRun>(edm::InputTag{"makeThingToBeDropped", "beginRun", "PROD"});
    }
    for(auto const& parent : expectedParents_) {
      mayConsume<edmtest::Thing>(edm::InputTag{parent,"event","PROD"});
    }
    if(expectedDroppedEvent1_.size() > droppedIndex1_) {
      consumes<edmtest::ThingWithIsEqual>(edm::InputTag{"makeThingToBeDropped1", "event", "PROD"});
    }
    consumes<edmtest::Thing>(edm::InputTag{"thingWithMergeProducer", "event", "PROD"});

    if(testAlias_){
      consumes<edmtest::Thing>(edm::InputTag{"aliasForThingToBeDropped2", "instance2"});
      consumes<edmtest::Thing>(edm::InputTag{"aliasForThingToBeDropped2", "instance2","PROD"});
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "beginRun", "PROD");
      consumes<edmtest::Thing,edm::InRun>(tag);
      consumes<edmtest::ThingWithMerge,edm::InRun>(tag);
      consumes<edmtest::ThingWithIsEqual,edm::InRun>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "beginRun");
      consumes<edmtest::Thing,edm::InRun>(tag);
      consumes<edmtest::ThingWithMerge,edm::InRun>(tag);
      consumes<edmtest::ThingWithIsEqual,edm::InRun>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "endRun", "PROD");
      consumes<edmtest::Thing,edm::InRun>(tag);
      consumes<edmtest::ThingWithMerge,edm::InRun>(tag);
      consumes<edmtest::ThingWithIsEqual,edm::InRun>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "endRun");
      consumes<edmtest::Thing,edm::InRun>(tag);
      consumes<edmtest::ThingWithMerge,edm::InRun>(tag);
      consumes<edmtest::ThingWithIsEqual,edm::InRun>(tag);
    }

    if(expectedDroppedEvent_.size() > 2) {
      edm::InputTag tag("makeThingToBeDropped", "endRun", "PROD");
      consumes<edmtest::ThingWithMerge,edm::InRun>(tag);
      consumes<edmtest::ThingWithIsEqual,edm::InRun>(tag);
    }
    
    if (testAlias_) {
      consumes<edmtest::Thing,edm::InRun>(edm::InputTag{"aliasForThingToBeDropped2", "endRun2"});
      edm::InputTag tag("aliasForThingToBeDropped2", "endRun2","PROD");
      consumes<edmtest::Thing,edm::InRun>(tag);
    }
  
    if(expectedDroppedEvent_.size() > 3) {
      edm::InputTag tag("makeThingToBeDropped", "beginLumi", "PROD");
      consumes<edmtest::ThingWithIsEqual,edm::InLumi>(tag);
    }
    {
      edm::InputTag tag("thingWithMergeProducer", "endLumi", "PROD");
      consumes<edmtest::Thing,edm::InLumi>(tag);
      consumes<edmtest::ThingWithMerge,edm::InLumi>(tag);
      consumes<edmtest::ThingWithIsEqual,edm::InLumi>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "endLumi");
      consumes<edmtest::Thing,edm::InLumi>(tag);
      consumes<edmtest::ThingWithMerge,edm::InLumi>(tag);
      consumes<edmtest::ThingWithIsEqual,edm::InLumi>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "beginLumi", "PROD");
      consumes<edmtest::Thing,edm::InLumi>(tag);
      consumes<edmtest::ThingWithMerge,edm::InLumi>(tag);
      consumes<edmtest::ThingWithIsEqual,edm::InLumi>(tag);
    }

    {
      edm::InputTag tag("thingWithMergeProducer", "beginLumi");
      consumes<edmtest::Thing,edm::InLumi>(tag);
      consumes<edmtest::ThingWithMerge,edm::InLumi>(tag);
      consumes<edmtest::ThingWithIsEqual,edm::InLumi>(tag);
    }

    if(expectedDroppedEvent_.size() > 4) {
      edm::InputTag tag("makeThingToBeDropped", "endLumi", "PROD");
      consumes<edmtest::ThingWithIsEqual,edm::InLumi>(tag);
      consumes<edmtest::ThingWithMerge,edm::InLumi>(tag);
    }

    if (testAlias_) {
      consumes<edmtest::Thing,edm::InLumi>(edm::InputTag{"aliasForThingToBeDropped2", "endLumi2"});
      edm::InputTag tag("aliasForThingToBeDropped2", "endLumi2","PROD");
      consumes<edmtest::Thing,edm::InLumi>(tag);
    }
  }

  // -----------------------------------------------------------------

  TestMergeResults::~TestMergeResults() {
  }

  // -----------------------------------------------------------------

  void TestMergeResults::analyze(edm::Event const& e,edm::EventSetup const&) {

    assert(e.processHistory().id() == e.processHistoryID());

    if(verbose_) edm::LogInfo("TestMergeResults") << "analyze";

    if(expectedDroppedEvent_.size() > 0) {
      edm::InputTag tag("makeThingToBeDropped", "event", "PROD");
      e.getByLabel(tag, h_thingWithIsEqual);
      assert(h_thingWithIsEqual->a == expectedDroppedEvent_[0]);

      e.getByLabel(tag, h_thingWithMerge);
      assert(h_thingWithMerge.isValid());
    }

    // This one is used to test the merging step when a specific product
    // has been dropped or not created in some of the input files.
    if(expectedDroppedEvent1_.size() > droppedIndex1_) {
      edm::InputTag tag("makeThingToBeDropped1", "event", "PROD");
      e.getByLabel(tag, h_thingWithIsEqual);
      if(expectedDroppedEvent1_[droppedIndex1_] == -1) {
        assert(!h_thingWithIsEqual.isValid());
      }
      else {
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
    if(parentIndex_ < expectedParents_.size()) {

      edm::InputTag tag("thingWithMergeProducer", "event", "PROD");
      e.getByLabel(tag, h_thing);
      std::string expectedParent = expectedParents_[parentIndex_];
      edm::BranchID actualParentBranchID = h_thing.provenance()->productProvenance()->parentage().parents()[0];

      // There ought to be a get that uses the BranchID as an argument, but
      // there is not at the moment so we get the Provenance first and use that
      // find the actual parent
      edm::Provenance prov = e.getProvenance(actualParentBranchID);
      assert(expectedParent == prov.moduleLabel());
      edm::InputTag tagparent(prov.moduleLabel(), prov.productInstanceName(), prov.processName());
      e.getByLabel(tagparent, h_thing);
      assert(h_thing->a == 11);
      ++parentIndex_;
    }

    if (testAlias_) {
      e.getByLabel("aliasForThingToBeDropped2", "instance2", h_thing);
      assert(h_thing->a == 11);
      edm::InputTag inputTag("aliasForThingToBeDropped2", "instance2","PROD");
      e.getByLabel(inputTag, h_thing);
      assert(h_thing->a == 11);

      edm::BranchID const& originalBranchID = h_thing.provenance()->branchDescription().originalBranchID();
      bool foundOriginalInRegistry = false;
      edm::Service<edm::ConstProductRegistry> reg;
      // Loop over provenance of products in registry.
      for (edm::ProductRegistry::ProductList::const_iterator it =  reg->productList().begin();
           it != reg->productList().end(); ++it) {
        edm::BranchDescription const& desc = it->second;
        if (desc.branchID() == originalBranchID) {
          foundOriginalInRegistry = true;
          break;
        }
      }
      assert(foundOriginalInRegistry);
    }
  }

  void TestMergeResults::beginRun(edm::Run const& run, edm::EventSetup const&) {

    if(verbose_) edm::LogInfo("TestMergeResults") << "beginRun";

    if(expectedDroppedEvent_.size() > 1) {
      edm::InputTag tagd("makeThingToBeDropped", "beginRun", "PROD");
      run.getByLabel(tagd, h_thingWithIsEqual);
      assert(h_thingWithIsEqual->a == expectedDroppedEvent_[1]);
    }
  }

  void TestMergeResults::endRun(edm::Run const& run, edm::EventSetup const&) {

    assert(run.processHistory().id() == run.processHistoryID());

    edm::ProcessHistory const& ph = run.processHistory();
    for (edm::ProcessHistory::const_iterator iter = ph.begin(), iEnd = ph.end();
         iter != iEnd; ++iter) {
      if (processHistoryIndex_ < expectedProcessHistoryInRuns_.size()) {
        assert(expectedProcessHistoryInRuns_[processHistoryIndex_] == iter->processName());
        ++processHistoryIndex_;
      }
    }

    if(verbose_) edm::LogInfo("TestMergeResults") << "endRun";

    std::vector<int> emptyDummy;

    edm::InputTag tag("thingWithMergeProducer", "endRun", "PROD");
    checkExpectedRunProducts(indexRun_, expectedEndRunProd_, tag, "endRun", run, expectedEndRunProdImproperlyMerged_);

    edm::InputTag tagnew("thingWithMergeProducer", "endRun");
    checkExpectedRunProducts(indexRun_, expectedEndRunNew_, tagnew, "endRun", run, emptyDummy);

    edm::InputTag tagb("thingWithMergeProducer", "beginRun", "PROD");
    checkExpectedRunProducts(indexRun_, expectedBeginRunProd_, tagb, "endRun", run, emptyDummy);

    edm::InputTag tagbnew("thingWithMergeProducer", "beginRun");
    checkExpectedRunProducts(indexRun_, expectedBeginRunNew_, tagbnew, "endRun", run, emptyDummy);

    if(expectedDroppedEvent_.size() > 2) {
      edm::InputTag tagd("makeThingToBeDropped", "endRun", "PROD");
      run.getByLabel(tagd, h_thingWithIsEqual);
      assert(h_thingWithIsEqual->a == expectedDroppedEvent_[2]);

      run.getByLabel(tagd, h_thingWithMerge);
      assert(!h_thingWithMerge.isValid());
    }

    if (testAlias_) {
      run.getByLabel("aliasForThingToBeDropped2", "endRun2", h_thing);
      assert(h_thing->a == 100001);
      edm::InputTag inputTag("aliasForThingToBeDropped2", "endRun2","PROD");
      run.getByLabel(inputTag, h_thing);
      assert(h_thing->a == 100001);

      edm::BranchID const& originalBranchID = h_thing.provenance()->branchDescription().originalBranchID();
      bool foundOriginalInRegistry = false;
      edm::Service<edm::ConstProductRegistry> reg;
      // Loop over provenance of products in registry.
      for (edm::ProductRegistry::ProductList::const_iterator it =  reg->productList().begin();
           it != reg->productList().end(); ++it) {
        edm::BranchDescription const& desc = it->second;
        if (desc.branchID() == originalBranchID) {
          foundOriginalInRegistry = true;
          break;
        }
      }
      assert(foundOriginalInRegistry);
    }

    indexRun_ += 3;
  }

  void TestMergeResults::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {

    if(verbose_) edm::LogInfo("TestMergeResults") << "beginLuminosityBlock";

    if(expectedDroppedEvent_.size() > 3) {
      edm::InputTag tagd("makeThingToBeDropped", "beginLumi", "PROD");
      lumi.getByLabel(tagd, h_thingWithIsEqual);
      assert(h_thingWithIsEqual->a == expectedDroppedEvent_[3]);
    }
  }

  void TestMergeResults::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {

    assert(lumi.processHistory().id() == lumi.processHistoryID());

    if(verbose_) edm::LogInfo("TestMergeResults") << "endLuminosityBlock";

    std::vector<int> emptyDummy;

    edm::InputTag tag("thingWithMergeProducer", "endLumi", "PROD");
    checkExpectedLumiProducts(indexLumi_, expectedEndLumiProd_, tag, "endLumi", lumi, expectedEndLumiProdImproperlyMerged_);

    edm::InputTag tagnew("thingWithMergeProducer", "endLumi");
    checkExpectedLumiProducts(indexLumi_, expectedEndLumiNew_, tagnew, "endLumi", lumi, emptyDummy);

    edm::InputTag tagb("thingWithMergeProducer", "beginLumi", "PROD");
    checkExpectedLumiProducts(indexLumi_, expectedBeginLumiProd_, tagb, "endLumi", lumi, emptyDummy);

    edm::InputTag tagbnew("thingWithMergeProducer", "beginLumi");
    checkExpectedLumiProducts(indexLumi_, expectedBeginLumiNew_, tagbnew, "endLumi", lumi, emptyDummy);

    if(expectedDroppedEvent_.size() > 4) {
      edm::InputTag tagd("makeThingToBeDropped", "endLumi", "PROD");
      lumi.getByLabel(tagd, h_thingWithIsEqual);
      assert(h_thingWithIsEqual->a == expectedDroppedEvent_[4]);

      lumi.getByLabel(tagd, h_thingWithMerge);
      assert(!h_thingWithMerge.isValid());
    }

    if (testAlias_) {
      lumi.getByLabel("aliasForThingToBeDropped2", "endLumi2", h_thing);
      assert(h_thing->a == 1001);
      edm::InputTag inputTag("aliasForThingToBeDropped2", "endLumi2","PROD");
      lumi.getByLabel(inputTag, h_thing);
      assert(h_thing->a == 1001);

      edm::BranchID const& originalBranchID = h_thing.provenance()->branchDescription().originalBranchID();
      bool foundOriginalInRegistry = false;
      edm::Service<edm::ConstProductRegistry> reg;
      // Loop over provenance of products in registry.
      for (edm::ProductRegistry::ProductList::const_iterator it =  reg->productList().begin();
           it != reg->productList().end(); ++it) {
        edm::BranchDescription const& desc = it->second;
        if (desc.branchID() == originalBranchID) {
          foundOriginalInRegistry = true;
          break;
        }
      }
      assert(foundOriginalInRegistry);
    }
    indexLumi_ += 3;
  }

  void TestMergeResults::respondToOpenInputFile(edm::FileBlock const& fb) {

    if(verbose_) edm::LogInfo("TestMergeResults") << "respondToOpenInputFile";

    if(!expectedInputFileNames_.empty()) {
      if(expectedInputFileNames_.size() <= static_cast<unsigned>(nRespondToOpenInputFile_) ||
          expectedInputFileNames_[nRespondToOpenInputFile_] != fb.fileName()) {
        std::cerr << "Error while testing merging of run/lumi products in TestMergeResults.cc\n"
                  << "Unexpected input filename, expected name = " << expectedInputFileNames_[nRespondToOpenInputFile_]
                  << "    actual name = " << fb.fileName() << std::endl;
        abort();
      }
    }
    ++nRespondToOpenInputFile_;
  }

  void TestMergeResults::respondToCloseInputFile(edm::FileBlock const&) {
    if(verbose_) edm::LogInfo("TestMergeResults") << "respondToCloseInputFile";
    ++nRespondToCloseInputFile_;
    ++droppedIndex1_;
  }

  void TestMergeResults::endJob() {
    if(verbose_) edm::LogInfo("TestMergeResults") << "endJob";

    if(expectedRespondToOpenInputFile_ > -1 && nRespondToOpenInputFile_ != expectedRespondToOpenInputFile_) {
      std::cerr << "Error while testing merging of run/lumi products in TestMergeResults.cc\n"
                << "Unexpected number of calls to the function respondToOpenInputFile" << std::endl;
      abort();
    }

    if(expectedRespondToCloseInputFile_ > -1 && nRespondToCloseInputFile_ != expectedRespondToCloseInputFile_) {
      std::cerr << "Error while testing merging of run/lumi products in TestMergeResults.cc\n"
                << "Unexpected number of calls to the function respondToCloseInputFile" << std::endl;
      abort();
    }
  }

  void
  TestMergeResults::checkExpectedRunProducts(unsigned int index,
                                             std::vector<int> const& expectedValues,
                                             edm::InputTag const& tag,
                                             char const* functionName,
                                             edm::Run const& run,
                                             std::vector<int> const& expectedValueImproperlyMerged) {

    if((index + 2) < expectedValues.size()) {

      int expected = expectedValues[index];
      if(expected != 0) {
        run.getByLabel(tag, h_thing);
        if(h_thing->a != expected) {
          abortWithMessage(functionName, "Thing", tag, expected, h_thing->a);
        }
        if (index < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index] != 0) != h_thing.provenance()->knownImproperlyMerged()) {
            abortWithMessage(functionName, "Thing", tag, 0, 0, true);
          }
        }
      }

      expected = expectedValues[index + 1];
      if(expected != 0) {
        run.getByLabel(tag, h_thingWithMerge);
        if(h_thingWithMerge->a != expected) {
          abortWithMessage(functionName, "ThingWithMerge", tag, expected, h_thingWithMerge->a);
        }
        if (index + 1 < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index + 1] != 0) != h_thingWithMerge.provenance()->knownImproperlyMerged()) {
            abortWithMessage(functionName, "ThingWithMerge", tag, 0, 0, true);
          }
        }
        if (!h_thingWithMerge.provenance()->branchDescription().isMergeable()) {
          std::cerr << "TestMergeResults::checkExpectedRunProducts isMergeable from BranchDescription returns\n"
                    << "unexpected value for ThingWithMerge type." << std::endl;
          abort();
        }
      }

      expected = expectedValues[index + 2];
      if(expected != 0) {
        run.getByLabel(tag, h_thingWithIsEqual);
        if(h_thingWithIsEqual->a != expected) {
          abortWithMessage(functionName, "ThingWithIsEqual", tag, expected, h_thingWithIsEqual->a);
        }
        if (index + 2 < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index + 2] != 0) != h_thingWithIsEqual.provenance()->knownImproperlyMerged()) {
            abortWithMessage(functionName, "ThingWithIsEqual", tag, 0, 0, true);
          }
        }
        if (h_thingWithIsEqual.provenance()->branchDescription().isMergeable()) {
          std::cerr << "TestMergeResults::checkExpectedRunProducts isMergeable from BranchDescription returns\n"
                    << "unexpected value for ThingWithIsEqual type." << std::endl;
          abort();
        }
      }
    }
  }

  void
  TestMergeResults::checkExpectedLumiProducts(unsigned int index,
                                              std::vector<int> const& expectedValues,
                                              edm::InputTag const& tag,
                                              char const* functionName,
                                              edm::LuminosityBlock const& lumi,
                                              std::vector<int> const& expectedValueImproperlyMerged) {

    if((index + 2) < expectedValues.size()) {

      int expected = expectedValues[index];
      if(expected != 0) {
        lumi.getByLabel(tag, h_thing);
        if(h_thing->a != expected) {
          abortWithMessage(functionName, "Thing", tag, expected, h_thing->a);
        }
        if (index < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index] != 0) != h_thing.provenance()->knownImproperlyMerged()) {
            abortWithMessage(functionName, "Thing", tag, 0, 0, true);
          }
        }
      }

      expected = expectedValues[index + 1];
      if(expected != 0) {
        lumi.getByLabel(tag, h_thingWithMerge);
        if(h_thingWithMerge->a != expected) {
          abortWithMessage(functionName, "ThingWithMerge", tag, expected, h_thingWithMerge->a);
        }
        if (index + 1 < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index + 1] != 0) != h_thingWithMerge.provenance()->knownImproperlyMerged()) {
            abortWithMessage(functionName, "ThingWithMerge", tag, 0, 0, true);
          }
        }
        if (!h_thingWithMerge.provenance()->branchDescription().isMergeable()) {
          std::cerr << "TestMergeResults::checkExpectedLumiProducts isMergeable from BranchDescription returns\n"
                    << "unexpected value for ThingWithMerge type." << std::endl;
          abort();
        }
      }

      expected = expectedValues[index + 2];
      if(expected != 0) {
        lumi.getByLabel(tag, h_thingWithIsEqual);
        if(h_thingWithIsEqual->a != expected) {
          abortWithMessage(functionName, "ThingWithIsEqual", tag, expected, h_thingWithIsEqual->a);
        }
        if (index + 2 < expectedValueImproperlyMerged.size()) {
          if ((expectedValueImproperlyMerged[index + 2] != 0) != h_thingWithIsEqual.provenance()->knownImproperlyMerged()) {
            abortWithMessage(functionName, "ThingWithIsEqual", tag, 0, 0, true);
          }
        }
        if (h_thingWithIsEqual.provenance()->branchDescription().isMergeable()) {
          std::cerr << "TestMergeResults::checkExpectedLumiProducts isMergeable from BranchDescription returns\n"
                    << "unexpected value for ThingWithIsEqual type." << std::endl;
          abort();
        }
      }
    }
  }

  void TestMergeResults::abortWithMessage(char const* whichFunction, char const* type, edm::InputTag const& tag,
                                          int expectedValue, int actualValue, bool unexpectedImproperlyMergedValue) const {
    std::cerr << "Error while testing merging of run/lumi products in TestMergeResults.cc\n"
              << "In function " << whichFunction << " looking for product of type " << type << "\n"
              << tag << std::endl;
    if (unexpectedImproperlyMergedValue) {
      std::cerr << "Unexpected value of knownImproperlyMerged from provenance" << std::endl;
    } else {
      std::cerr << "Expected value = " << expectedValue << " actual value = " << actualValue << std::endl;
    }
    abort();
  }
}

using edmtest::TestMergeResults;

DEFINE_FWK_MODULE(TestMergeResults);
