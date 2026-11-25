
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/Provenance/interface/ProductDescriptionFwd.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleLabelMatch.h"
#include "FWCore/Framework/interface/ProcessMatch.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <string>
#include <vector>

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace edmtest {

  class TestGetterOfProducts : public edm::stream::EDFilter<> {
  public:
    explicit TestGetterOfProducts(edm::ParameterSet const&);
    ~TestGetterOfProducts() override;
    bool filter(edm::Event&, edm::EventSetup const&) override;

  private:
    std::string processName_;
    std::vector<std::string> expectedInputTagLabels_;
    std::vector<std::string> expectedLabelsAfterGet_;
    edm::GetterOfProducts<Thing> getterOfProducts_;
  };

  TestGetterOfProducts::TestGetterOfProducts(edm::ParameterSet const& par)
      : processName_(par.getParameter<std::string>("processName")),
        expectedInputTagLabels_(par.getParameter<std::vector<std::string> >("expectedInputTagLabels")),
        expectedLabelsAfterGet_(par.getParameter<std::vector<std::string> >("expectedLabelsAfterGet")),
        getterOfProducts_(edm::ProcessMatch(processName_), this) {
    callWhenNewProductsRegistered(getterOfProducts_);
  }

  TestGetterOfProducts::~TestGetterOfProducts() {}

  bool TestGetterOfProducts::filter(edm::Event& event, edm::EventSetup const&) {
    // These are the InputTag's of the products the getter will look for.
    // These are the matching products in the ProductRegistry, but they
    // may or may not be in a particular Event.
    auto const& tokens = getterOfProducts_.tokens();

    if (expectedInputTagLabels_.size() != tokens.size()) {
      std::cerr << "Expected number of InputTag's differs from actual number.\n"
                << "In function TestGetterOfProducts::filter\n"
                << "Expected value = " << expectedInputTagLabels_.size() << " actual value = " << tokens.size()
                << std::endl;
      abort();
    }

    std::vector<std::string>::const_iterator iter = expectedInputTagLabels_.begin();
    for (auto const& token : tokens) {
      edm::EDConsumerBase::Labels labels;
      labelsForToken(token, labels);
      if (*iter != labels.module) {
        throw cms::Exception("Failed Test")
            << "Expected InputTag differs from actual InputTag.\n"
            << "In function TestGetterOfProducts::filter\n"
            << "Expected value = " << *iter << " actual value = " << labels.module << std::endl;
      }
      ++iter;
    }

    // A handle is added to this vector for each product that is found in the event.
    std::vector<edm::Handle<Thing> > handles;
    getterOfProducts_.fillHandles(event, handles);

    if (expectedLabelsAfterGet_.size() != handles.size()) {
      throw cms::Exception("Failed Test")
          << "Expected number of products gotten differs from actual number.\n"
          << "In function TestGetterOfProducts::filter\n"
          << "Expected value = " << expectedLabelsAfterGet_.size() << " actual value = " << handles.size() << std::endl;
    }

    iter = expectedLabelsAfterGet_.begin();
    for (auto const& handle : handles) {
      if (handle.provenance()->moduleLabel() != *iter) {
        throw cms::Exception("Failed Test")
            << "Expected module label after get differs from actual module label.\n"
            << "In function TestGetterOfProducts::filter\n"
            << "Expected value = " << *iter << " actual value = " << handle.provenance()->moduleLabel() << std::endl;
        break;
      }
      ++iter;
    }

    return true;
  }

  // -----------------------------------------------------------------------------------------------------------

  // Very similar to the above class with these differences.
  // 1. The is an EDAnalyzer instead of a EDFilter
  // 2. It gets multiple types of products so there are
  // two GetterOfProducts objects.
  // 3. Initialize one of the GetterOfProducts later in the
  // constructor.
  // 4. It can be configured to get things from the Run, LuminosityBlock,
  // or Event.
  class TestGetterOfProductsA : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
  public:
    explicit TestGetterOfProductsA(edm::ParameterSet const&);
    ~TestGetterOfProductsA() override;
    void analyze(edm::Event const&, edm::EventSetup const&) override;
    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void endRun(edm::Run const&, edm::EventSetup const&) override;

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}
    void beginRun(edm::Run const&, edm::EventSetup const&) override {}

  private:
    std::string processName_;
    edm::BranchType branchType_;
    std::vector<std::string> expectedInputTagLabels_;
    std::vector<std::string> expectedLabelsAfterGet_;
    unsigned expectedNumberOfThingsWithLabelA_;
    edm::GetterOfProducts<IntProduct> getterOfIntProducts_;
    edm::GetterOfProducts<Thing, edm::Transition::BeginRun> getterOfProductsRun_;
    edm::GetterOfProducts<Thing, edm::Transition::BeginLuminosityBlock> getterOfProductsLumi_;
    edm::GetterOfProducts<Thing> getterOfProductsEvent_;
    edm::GetterOfProducts<Thing, edm::Transition::BeginRun> getterUsingLabelRun_;
    edm::GetterOfProducts<Thing, edm::Transition::BeginLuminosityBlock> getterUsingLabelLumi_;
    edm::GetterOfProducts<Thing> getterUsingLabelEvent_;
  };

  TestGetterOfProductsA::TestGetterOfProductsA(edm::ParameterSet const& par)
      : processName_(par.getParameter<std::string>("processName")),
        branchType_(edm::InEvent),
        expectedInputTagLabels_(par.getParameter<std::vector<std::string> >("expectedInputTagLabels")),
        expectedLabelsAfterGet_(par.getParameter<std::vector<std::string> >("expectedLabelsAfterGet")),
        expectedNumberOfThingsWithLabelA_(par.getParameter<unsigned>("expectedNumberOfThingsWithLabelA")),
        getterOfIntProducts_(),
        getterOfProductsRun_(),
        getterOfProductsLumi_(),
        getterOfProductsEvent_(),
        getterUsingLabelRun_(),
        getterUsingLabelLumi_(),
        getterUsingLabelEvent_() {
    int bt = par.getParameter<int>("branchType");
    getterOfIntProducts_ = edm::GetterOfProducts<IntProduct>(edm::ProcessMatch(processName_), this);

    if (bt == 1) {
      branchType_ = edm::InLumi;
      getterOfProductsLumi_ = edm::GetterOfProducts<Thing, edm::Transition::BeginLuminosityBlock>(
          edm::ProcessMatch(processName_), this, branchType_);
      getterUsingLabelLumi_ = edm::GetterOfProducts<Thing, edm::Transition::BeginLuminosityBlock>(
          edm::ModuleLabelMatch("A"), this, branchType_);
      callWhenNewProductsRegistered([this](edm::ProductDescription const& bd) {
        getterOfIntProducts_(bd);
        getterOfProductsLumi_(bd);
        getterUsingLabelLumi_(bd);
      });
    } else if (bt == 2) {
      branchType_ = edm::InRun;
      getterOfProductsRun_ =
          edm::GetterOfProducts<Thing, edm::Transition::BeginRun>(edm::ProcessMatch(processName_), this, branchType_);
      getterUsingLabelRun_ =
          edm::GetterOfProducts<Thing, edm::Transition::BeginRun>(edm::ModuleLabelMatch("A"), this, branchType_);
      callWhenNewProductsRegistered([this](edm::ProductDescription const& bd) {
        getterOfIntProducts_(bd);
        getterOfProductsRun_(bd);
        getterUsingLabelRun_(bd);
      });
    } else {
      getterOfProductsEvent_ = edm::GetterOfProducts<Thing>(edm::ProcessMatch(processName_), this, branchType_);
      getterUsingLabelEvent_ = edm::GetterOfProducts<Thing>(edm::ModuleLabelMatch("A"), this, branchType_);
      callWhenNewProductsRegistered([this](edm::ProductDescription const& bd) {
        getterOfIntProducts_(bd);
        getterOfProductsEvent_(bd);
        getterUsingLabelEvent_(bd);
      });
    }
  }

  TestGetterOfProductsA::~TestGetterOfProductsA() {}

  void TestGetterOfProductsA::analyze(edm::Event const& event, edm::EventSetup const&) {
    if (branchType_ != edm::InEvent)
      return;

    auto const& tokens = getterOfProductsEvent_.tokens();

    if (expectedInputTagLabels_.size() != tokens.size()) {
      throw cms::Exception("Failed Test")
          << "Expected number of InputTag's differs from actual number.\n"
          << "In function TestGetterOfProducts::filter\n"
          << "Expected value = " << expectedInputTagLabels_.size() << " actual value = " << tokens.size() << std::endl;
    }

    std::vector<std::string>::const_iterator iter = expectedInputTagLabels_.begin();
    for (auto const& token : tokens) {
      edm::EDConsumerBase::Labels labels;
      labelsForToken(token, labels);
      if (*iter != labels.module) {
        throw cms::Exception("Failed Test")
            << "Expected InputTag differs from actual InputTag.\n"
            << "In function TestGetterOfProducts::filter\n"
            << "Expected value = " << *iter << " actual value = " << labels.module << std::endl;
      }
      ++iter;
    }

    std::vector<edm::Handle<Thing> > handles;
    getterOfProductsEvent_.fillHandles(event, handles);

    if (expectedLabelsAfterGet_.size() != handles.size()) {
      throw cms::Exception("Failed Test")
          << "Expected number of products gotten differs from actual number.\n"
          << "In function TestGetterOfProducts::filter\n"
          << "Expected value = " << expectedLabelsAfterGet_.size() << " actual value = " << handles.size() << std::endl;
    }

    iter = expectedLabelsAfterGet_.begin();
    for (auto const& handle : handles) {
      if (handle.provenance()->moduleLabel() != *iter) {
        throw cms::Exception("Failed Test")
            << "Expected module label after get differs from actual module label.\n"
            << "In function TestGetterOfProducts::filter\n"
            << "Expected value = " << *iter << " actual value = " << handle.provenance()->moduleLabel() << std::endl;
        break;
      }
      ++iter;
    }

    getterUsingLabelEvent_.fillHandles(event, handles);

    if (expectedNumberOfThingsWithLabelA_ != handles.size()) {
      throw cms::Exception("Failed Test")
          << "Expected number of products with module label A gotten differs from actual number.\n"
          << "In function TestGetterOfProducts::filter\n"
          << "Expected value = " << expectedNumberOfThingsWithLabelA_ << " actual value = " << handles.size()
          << std::endl;
    }

    iter = expectedLabelsAfterGet_.begin();
    for (auto const& handle : handles) {
      if (handle.provenance()->moduleLabel() != std::string("A")) {
        throw cms::Exception("Failed Test") << "Expected module label after get differs from actual module label.\n"
                                            << "In function TestGetterOfProducts::filter\n"
                                            << "For this get a module label \"A\" is always expected "
                                            << " actual value = " << handle.provenance()->moduleLabel() << std::endl;
        break;
      }
      ++iter;
    }
  }

  void TestGetterOfProductsA::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
    if (branchType_ != edm::InLumi)
      return;

    auto const& tokens = getterOfProductsLumi_.tokens();

    if (expectedInputTagLabels_.size() != tokens.size()) {
      throw cms::Exception("Failed Test")
          << "Expected number of InputTag's differs from actual number.\n"
          << "In function TestGetterOfProducts::endLuminosityBlock\n"
          << "Expected value = " << expectedInputTagLabels_.size() << " actual value = " << tokens.size() << std::endl;
    }

    std::vector<std::string>::const_iterator iter = expectedInputTagLabels_.begin();
    for (auto const& token : tokens) {
      edm::EDConsumerBase::Labels labels;
      labelsForToken(token, labels);

      if (*iter != labels.module) {
        throw cms::Exception("Failed Test")
            << "Expected InputTag differs from actual InputTag.\n"
            << "In function TestGetterOfProducts::endLuminosityBlock\n"
            << "Expected value = " << *iter << " actual value = " << labels.module << std::endl;
      }
      ++iter;
    }

    std::vector<edm::Handle<Thing> > handles;
    getterOfProductsLumi_.fillHandles(lumi, handles);

    if (expectedLabelsAfterGet_.size() != handles.size()) {
      throw cms::Exception("Failed Test")
          << "Expected number of products gotten differs from actual number.\n"
          << "In function TestGetterOfProducts::endLuminosityBlock\n"
          << "Expected value = " << expectedLabelsAfterGet_.size() << " actual value = " << handles.size() << std::endl;
    }

    iter = expectedLabelsAfterGet_.begin();
    for (auto const& handle : handles) {
      if (handle.provenance()->moduleLabel() != *iter) {
        throw cms::Exception("Failed Test")
            << "Expected module label after get differs from actual module label.\n"
            << "In function TestGetterOfProducts::endLuminosityBlock\n"
            << "Expected value = " << *iter << " actual value = " << handle.provenance()->moduleLabel() << std::endl;
        break;
      }
      ++iter;
    }

    getterUsingLabelLumi_.fillHandles(lumi, handles);

    if (expectedNumberOfThingsWithLabelA_ != handles.size()) {
      throw cms::Exception("Failed Test")
          << "Expected number of products with module label A gotten differs from actual number.\n"
          << "In function TestGetterOfProducts::endLuminosityBlock\n"
          << "Expected value = " << expectedNumberOfThingsWithLabelA_ << " actual value = " << handles.size()
          << std::endl;
    }

    iter = expectedLabelsAfterGet_.begin();
    for (auto const& handle : handles) {
      if (handle.provenance()->moduleLabel() != std::string("A")) {
        throw cms::Exception("Failed Test") << "Expected module label after get differs from actual module label.\n"
                                            << "In function TestGetterOfProducts::endLuminosityBlock\n"
                                            << "For this get a module label \"A\" is always expected "
                                            << " actual value = " << handle.provenance()->moduleLabel() << std::endl;
        break;
      }
      ++iter;
    }
  }

  void TestGetterOfProductsA::endRun(edm::Run const& run, edm::EventSetup const&) {
    if (branchType_ != edm::InRun)
      return;

    auto const& tokens = getterOfProductsRun_.tokens();

    if (expectedInputTagLabels_.size() != tokens.size()) {
      throw cms::Exception("Failed Test")
          << "Expected number of InputTag's differs from actual number.\n"
          << "In function TestGetterOfProducts::endRun\n"
          << "Expected value = " << expectedInputTagLabels_.size() << " actual value = " << tokens.size() << std::endl;
    }

    std::vector<std::string>::const_iterator iter = expectedInputTagLabels_.begin();
    for (auto const& token : tokens) {
      edm::EDConsumerBase::Labels labels;
      labelsForToken(token, labels);
      if (*iter != labels.module) {
        throw cms::Exception("Failed Test")
            << "Expected InputTag differs from actual InputTag.\n"
            << "In function TestGetterOfProducts::endRun\n"
            << "Expected value = " << *iter << " actual value = " << labels.module << std::endl;
      }
      ++iter;
    }

    std::vector<edm::Handle<Thing> > handles;
    getterOfProductsRun_.fillHandles(run, handles);

    if (expectedLabelsAfterGet_.size() != handles.size()) {
      throw cms::Exception("Failed Test")
          << "Expected number of products gotten differs from actual number.\n"
          << "In function TestGetterOfProducts::endRun\n"
          << "Expected value = " << expectedLabelsAfterGet_.size() << " actual value = " << handles.size() << std::endl;
    }

    iter = expectedLabelsAfterGet_.begin();
    for (auto const& handle : handles) {
      if (handle.provenance()->moduleLabel() != *iter) {
        throw cms::Exception("Failed Test")
            << "Expected module label after get differs from actual module label.\n"
            << "In function TestGetterOfProducts::endRun\n"
            << "Expected value = " << *iter << " actual value = " << handle.provenance()->moduleLabel() << std::endl;
        break;
      }
      ++iter;
    }

    getterUsingLabelRun_.fillHandles(run, handles);

    if (expectedNumberOfThingsWithLabelA_ != handles.size()) {
      throw cms::Exception("Failed Test")
          << "Expected number of products with module label A gotten differs from actual number.\n"
          << "In function TestGetterOfProducts::endRun\n"
          << "Expected value = " << expectedNumberOfThingsWithLabelA_ << " actual value = " << handles.size()
          << std::endl;
    }

    iter = expectedLabelsAfterGet_.begin();
    for (auto const& handle : handles) {
      if (handle.provenance()->moduleLabel() != std::string("A")) {
        throw cms::Exception("Failed Test") << "Expected module label after get differs from actual module label.\n"
                                            << "In function TestGetterOfProducts::endRun\n"
                                            << "For this get a module label \"A\" is always expected "
                                            << " actual value = " << handle.provenance()->moduleLabel() << std::endl;
        break;
      }
      ++iter;
    }
  }
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_MODULE(TestGetterOfProducts);
DEFINE_FWK_MODULE(TestGetterOfProductsA);
