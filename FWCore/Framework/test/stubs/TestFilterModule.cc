
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>

using namespace edm;

namespace edm {
  class ModuleCallingContext;
}

namespace edmtest {

  class TestResultAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit TestResultAnalyzer(edm::ParameterSet const&);
    ~TestResultAnalyzer() override;

    void analyze(edm::Event const& e, edm::EventSetup const& c) override;
    void endJob() override;

  private:
    int passed_;
    int failed_;
    std::string name_;
    int numbits_;
  };

  class TestContextAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit TestContextAnalyzer(edm::ParameterSet const&);

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& c) const override;
    std::string expected_pathname_;     // if empty, we don't know
    std::string expected_modulelabel_;  // if empty, we don't know
  };

  // -------

  class TestFilterModule : public edm::one::EDFilter<> {
  public:
    explicit TestFilterModule(edm::ParameterSet const&);
    ~TestFilterModule() override;

    bool filter(edm::Event& e, edm::EventSetup const& c) override;
    void endJob() override;

  private:
    int count_;
    int accept_rate_;  // how many out of 100 will be accepted?
    bool onlyOne_;
  };

  // -------

  class SewerModule : public edm::one::OutputModule<> {
  public:
    explicit SewerModule(edm::ParameterSet const&);
    ~SewerModule() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void write(edm::EventForOutput const& e) override;
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override {}
    void writeRun(edm::RunForOutput const&) override {}
    void endJob() override;

    std::string name_;
    int num_pass_;
    int total_;
  };

  // -----------------------------------------------------------------

  TestResultAnalyzer::TestResultAnalyzer(edm::ParameterSet const& ps)
      : passed_(),
        failed_(),
        name_(ps.getUntrackedParameter<std::string>("name", "DEFAULT")),
        numbits_(ps.getUntrackedParameter<int>("numbits", -1)) {
    consumesMany<edm::TriggerResults>();
  }

  TestResultAnalyzer::~TestResultAnalyzer() {}

  void TestResultAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {
    typedef std::vector<edm::Handle<edm::TriggerResults> > Trig;
    Trig prod;
    e.getManyByType(prod);

    if (prod.empty())
      return;
    if (prod.size() > 1) {
      std::cerr << "More than one trigger result in the event, using first one" << std::endl;
    }

    if (prod[0]->accept())
      ++passed_;
    else
      ++failed_;

    if (numbits_ < 0)
      return;

    unsigned int numbits = numbits_;
    if (numbits != prod[0]->size()) {
      std::cerr << "TestResultAnalyzer named: " << name_ << " should have " << numbits << ", got " << prod[0]->size()
                << " in TriggerResults\n";
      abort();
    }
  }

  void TestResultAnalyzer::endJob() {
    std::cerr << "TESTRESULTANALYZER " << name_ << ": "
              << "passed=" << passed_ << " failed=" << failed_ << "\n";
  }

  // ---------

  TestContextAnalyzer::TestContextAnalyzer(edm::ParameterSet const& ps)
      : expected_pathname_(ps.getUntrackedParameter<std::string>("pathname", "")),
        expected_modulelabel_(ps.getUntrackedParameter<std::string>("modlabel", "")) {}

  void TestContextAnalyzer::analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const&) const {
    assert(e.moduleCallingContext()->moduleDescription()->moduleLabel() == moduleDescription().moduleLabel());

    if (!expected_pathname_.empty()) {
      assert(expected_pathname_ == e.moduleCallingContext()->placeInPathContext()->pathContext()->pathName());
    }

    if (!expected_modulelabel_.empty()) {
      assert(expected_modulelabel_ == moduleDescription().moduleLabel());
    }
  }

  // ---------

  TestFilterModule::TestFilterModule(edm::ParameterSet const& ps)
      : count_(),
        accept_rate_(ps.getUntrackedParameter<int>("acceptValue", 1)),
        onlyOne_(ps.getUntrackedParameter<bool>("onlyOne", false)) {}

  TestFilterModule::~TestFilterModule() {}

  bool TestFilterModule::filter(edm::Event& e, edm::EventSetup const&) {
    assert(e.moduleCallingContext()->moduleDescription()->moduleLabel() == moduleDescription().moduleLabel());

    ++count_;
    if (onlyOne_)
      return count_ % accept_rate_ == 0;
    else
      return count_ % 100 <= accept_rate_;
  }

  void TestFilterModule::endJob() {}

  // ---------

  SewerModule::SewerModule(edm::ParameterSet const& ps)
      : edm::one::OutputModuleBase::OutputModuleBase(ps),
        edm::one::OutputModule<>(ps),
        name_(ps.getParameter<std::string>("name")),
        num_pass_(ps.getParameter<int>("shouldPass")),
        total_() {}

  SewerModule::~SewerModule() {}

  void SewerModule::write(edm::EventForOutput const&) { ++total_; }

  void SewerModule::endJob() {
    std::cerr << "SEWERMODULE " << name_ << ": should pass " << num_pass_ << ", did pass " << total_ << "\n";

    if (total_ != num_pass_) {
      std::cerr << "number passed should be " << num_pass_ << ", but got " << total_ << "\n";
      abort();
    }
  }

  void SewerModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Tracks number of times the write method is called.");
    desc.add<std::string>("name")->setComment("name used in printout");
    desc.add<int>("shouldPass")->setComment("number of times write should be called");
    edm::one::OutputModule<>::fillDescription(desc, std::vector<std::string>(1U, std::string("drop *")));
    descriptions.add("sewerModule", desc);
  }

}  // namespace edmtest

using edmtest::SewerModule;
using edmtest::TestContextAnalyzer;
using edmtest::TestFilterModule;
using edmtest::TestResultAnalyzer;

DEFINE_FWK_MODULE(TestFilterModule);
DEFINE_FWK_MODULE(TestResultAnalyzer);
DEFINE_FWK_MODULE(SewerModule);
DEFINE_FWK_MODULE(TestContextAnalyzer);
