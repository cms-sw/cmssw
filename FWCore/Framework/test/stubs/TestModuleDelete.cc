#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <string>
#include <vector>

namespace {
  // No need to protect, all access is serialized by the framework
  std::vector<std::string> g_constructed;
  std::vector<std::string> g_destructed;

  std::vector<std::string> g_lumi_constructed;
  std::vector<std::string> g_lumi_destructed;

  std::vector<std::string> g_run_constructed;
  std::vector<std::string> g_run_destructed;

  std::vector<std::string> g_process_constructed;
  std::vector<std::string> g_process_destructed;
}  // namespace

namespace edmtest {
  // Similar to FailingProducer, but set a flag when deleted
  class TestModuleDeleteProducer : public edm::global::EDProducer<> {
  public:
    explicit TestModuleDeleteProducer(edm::ParameterSet const& p)
        : moduleLabel_{p.getParameter<std::string>("@module_label")} {
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginProcess")) {
        consumes<IntProduct, edm::InProcess>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginRun")) {
        consumes<IntProduct, edm::InRun>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginLumi")) {
        consumes<IntProduct, edm::InLumi>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcEvent")) {
        consumes<IntProduct>(tag);
      }
      produces<IntProduct>();
      g_constructed.push_back(moduleLabel_);
    }
    ~TestModuleDeleteProducer() override { g_destructed.push_back(moduleLabel_); }
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginProcess", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginRun", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginLumi", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEvent", std::vector<edm::InputTag>{});
      descriptions.addDefault(desc);
    }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override {
      throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
    }

  private:
    std::string moduleLabel_;
  };

  class TestModuleDeleteInLumiProducer : public edm::global::EDProducer<edm::BeginLuminosityBlockProducer> {
  public:
    explicit TestModuleDeleteInLumiProducer(edm::ParameterSet const& p)
        : moduleLabel_{p.getParameter<std::string>("@module_label")} {
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginProcess")) {
        consumes<IntProduct, edm::InProcess>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginRun")) {
        consumes<IntProduct, edm::InRun>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginLumi")) {
        consumes<IntProduct, edm::InLumi>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcEvent")) {
        consumes<IntProduct, edm::InRun>(tag);
      }
      produces<IntProduct, edm::Transition::BeginLuminosityBlock>();
      g_lumi_constructed.push_back(moduleLabel_);
    }
    ~TestModuleDeleteInLumiProducer() override { g_lumi_destructed.push_back(moduleLabel_); }
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginProcess", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginRun", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginLumi", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEvent", std::vector<edm::InputTag>{});
      descriptions.addDefault(desc);
    }
    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override {
      throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
    }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override {
      throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
    }

  private:
    std::string moduleLabel_;
  };

  class TestModuleDeleteInRunProducer : public edm::global::EDProducer<edm::BeginRunProducer> {
  public:
    explicit TestModuleDeleteInRunProducer(edm::ParameterSet const& p)
        : moduleLabel_{p.getParameter<std::string>("@module_label")} {
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginProcess")) {
        consumes<IntProduct, edm::InProcess>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginRun")) {
        consumes<IntProduct, edm::InRun>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginLumi")) {
        consumes<IntProduct, edm::InLumi>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcEvent")) {
        consumes<IntProduct, edm::InRun>(tag);
      }
      produces<IntProduct, edm::Transition::BeginRun>();
      g_run_constructed.push_back(moduleLabel_);
    }
    ~TestModuleDeleteInRunProducer() override { g_run_destructed.push_back(moduleLabel_); }
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginProcess", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginRun", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginLumi", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEvent", std::vector<edm::InputTag>{});
      descriptions.addDefault(desc);
    }
    void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const override {
      throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
    }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override {
      throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
    }

  private:
    std::string moduleLabel_;
  };

  class TestModuleDeleteInProcessProducer : public edm::global::EDProducer<edm::BeginProcessBlockProducer> {
  public:
    explicit TestModuleDeleteInProcessProducer(edm::ParameterSet const& p)
        : moduleLabel_{p.getParameter<std::string>("@module_label")} {
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginRun")) {
        consumes<IntProduct, edm::InRun>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginLumi")) {
        consumes<IntProduct, edm::InLumi>(tag);
      }
      for (const auto& tag : p.getUntrackedParameter<std::vector<edm::InputTag>>("srcEvent")) {
        consumes<IntProduct, edm::InRun>(tag);
      }
      produces<IntProduct, edm::Transition::BeginProcessBlock>();
      g_process_constructed.push_back(moduleLabel_);
    }
    ~TestModuleDeleteInProcessProducer() override { g_process_destructed.push_back(moduleLabel_); }
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginRun", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginLumi", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEvent", std::vector<edm::InputTag>{});
      descriptions.addDefault(desc);
    }
    void beginProcessBlockProduce(edm::ProcessBlock&) const override {
      throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
    }
    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override {
      throw edm::Exception(edm::errors::NotFound) << "Intentional 'NotFound' exception for testing purposes\n";
    }

  private:
    std::string moduleLabel_;
  };

  class TestModuleDeleteAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit TestModuleDeleteAnalyzer(edm::ParameterSet const& /*p*/){};

    void beginJob() override {
      if (g_constructed.size() == 0) {
        throw cms::Exception("LogicError")
            << "No TestModuleDeleteProducer modules constructed in this job, the test is meaningless";
      }

      auto formatException =
          [](std::vector<std::string>& constructed, std::vector<std::string>& destructed, cms::Exception& ex) {
            ex << " Modules constructed but not destructed:\n";
            auto newEnd = std::remove_if(constructed.begin(), constructed.end(), [&](const std::string& label) {
              auto found = std::find(destructed.begin(), destructed.end(), label);
              if (found != destructed.end()) {
                destructed.erase(found);
                return true;
              }
              return false;
            });
            constructed.erase(newEnd, constructed.end());
            for (const auto& lab : constructed) {
              ex << " " << lab << "\n";
            }
          };

      if (g_constructed.size() != g_destructed.size()) {
        cms::Exception ex("Assert");
        ex << "Number of TestModuleDeleteProducer constructors " << g_constructed.size()
           << " differs from the number of destructors " << g_destructed.size() << ".";
        formatException(g_constructed, g_destructed, ex);
        throw ex;
      }

      if (g_lumi_constructed.size() == 0) {
        throw cms::Exception("LogicError")
            << "No TestModuleDeleteInLumiProducer modules constructed in this job, the test is meaningless";
      }
      if (g_lumi_constructed.size() != g_lumi_destructed.size()) {
        cms::Exception ex("Assert");
        ex << "Number of TestModuleDeleteInLumiProducer constructors " << g_lumi_constructed.size()
           << " differs from the number of destructors " << g_lumi_destructed.size() << ".";
        formatException(g_lumi_constructed, g_lumi_destructed, ex);
        throw ex;
      }

      if (g_run_constructed.size() == 0) {
        throw cms::Exception("LogicError")
            << "No TestModuleDeleteInRunProducer modules constructed in this job, the test is meaningless";
      }
      if (g_run_constructed.size() != g_run_destructed.size()) {
        cms::Exception ex("Assert");
        ex << "Number of TestModuleDeleteInRunProducer constructors " << g_run_constructed.size()
           << " differs from the number of destructors " << g_run_destructed.size() << ".";
        formatException(g_run_constructed, g_run_destructed, ex);
        throw ex;
      }

      if (g_process_constructed.size() == 0) {
        throw cms::Exception("LogicError")
            << "No TestModuleDeleteInProcessProducer modules constructed in this job, the test is meaningless";
      }
      if (g_process_constructed.size() != g_process_destructed.size()) {
        cms::Exception ex("Assert");
        ex << "Number of TestModuleDeleteInProcessProducer constructors " << g_process_constructed.size()
           << " differs from the number of destructors " << g_process_destructed.size() << ".";
        formatException(g_process_constructed, g_process_destructed, ex);
        throw ex;
      }
    }

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& c) const override {}
  };
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::TestModuleDeleteProducer);
DEFINE_FWK_MODULE(edmtest::TestModuleDeleteInLumiProducer);
DEFINE_FWK_MODULE(edmtest::TestModuleDeleteInRunProducer);
DEFINE_FWK_MODULE(edmtest::TestModuleDeleteInProcessProducer);
DEFINE_FWK_MODULE(edmtest::TestModuleDeleteAnalyzer);
