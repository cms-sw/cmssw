#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {
  void require(bool condition, char const* message) {
    if (!condition) {
      throw cms::Exception("FileBlockTransitionTest") << message;
    }
  }
}  // namespace

namespace edmtest::pool {
  // Produces event-dependent input values for the readback checks.
  class TransitionValueProducer : public edm::stream::EDProducer<> {
  public:
    explicit TransitionValueProducer(edm::ParameterSet const&) : token_(produces<edmtest::IntProduct>()) {}
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addDefault(desc);
    }

  private:
    void produce(edm::Event& event, edm::EventSetup const&) override {
      event.emplace(token_, static_cast<int>(event.id().event()) * 11);
    }
    edm::EDPutTokenT<edmtest::IntProduct> token_;
  };

  // Checks retained values and merged dependency edges, including the edge
  // present only in the middle input, for every event in the output file.
  class TransitionValueChecker : public edm::one::OutputModule<edm::WatchInputFiles> {
  public:
    explicit TransitionValueChecker(edm::ParameterSet const& ps)
        : edm::one::OutputModuleBase(ps),
          edm::one::OutputModule<edm::WatchInputFiles>(ps),
          token_(consumes<edmtest::IntProduct>(ps.getParameter<edm::InputTag>("src"))),
          parentToken_(consumes<edmtest::IntProduct>(ps.getParameter<edm::InputTag>("parent"))),
          specialToken_(consumes<edmtest::IntProduct>(ps.getParameter<edm::InputTag>("special"))),
          specialParentToken_(consumes<edmtest::IntProduct>(ps.getParameter<edm::InputTag>("specialParent"))) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      edm::one::OutputModule<edm::WatchInputFiles>::fillDescription(desc);
      desc.add<edm::InputTag>("src");
      desc.add<edm::InputTag>("parent");
      desc.add<edm::InputTag>("special");
      desc.add<edm::InputTag>("specialParent");
      descriptions.addDefault(desc);
    }

  private:
    void respondToOpenInputFile(edm::FileBlock const& fb) override {
      require(fb.tree() && !fb.productDependencies().childLookup().empty(), "readback dependencies are empty");
      dependencies_ = fb.productDependencies();
      ++files_;
    }
    void respondToCloseInputFile(edm::FileBlock const&) override {}

    void write(edm::EventForOutput const& event) override {
      require(event.id().event() >= 1 && event.id().event() <= 6, "unexpected event ID");
      auto child = event.getHandle(token_);
      auto parent = event.getHandle(parentToken_);
      auto special = event.getHandle(specialToken_);
      auto specialParent = event.getHandle(specialParentToken_);
      require(child.isValid() && parent.isValid() && special.isValid() && specialParent.isValid(),
              "retained products are missing");
      require(child->value == static_cast<int>(event.id().event()) * 11 && child->value == parent->value,
              "wrong retained product value");
      auto const& lookup = dependencies_.childLookup();
      auto found = lookup.find(parent.provenance()->branchID());
      require(found != lookup.end() && found->second.count(child.provenance()->branchID()),
              "output dependency graph does not contain parent -> child");
      auto specialFound = lookup.find(specialParent.provenance()->branchID());
      require(specialFound != lookup.end() && specialFound->second.count(special.provenance()->branchID()),
              "output dependency graph lost the middle-only edge");
      require(specialParent->value == parent->value &&
                  special->value == (event.id().event() == 3 ? specialParent->value : 0),
              "wrong middle-only dependency product value");
      ++count_;
    }
    void writeRun(edm::RunForOutput const&) override {}
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override {}
    void endJob() override { require(count_ == 6 && files_ == 1, "missing events or output files"); }

    edm::EDGetTokenT<edmtest::IntProduct> token_;
    edm::EDGetTokenT<edmtest::IntProduct> parentToken_;
    edm::EDGetTokenT<edmtest::IntProduct> specialToken_;
    edm::EDGetTokenT<edmtest::IntProduct> specialParentToken_;
    edm::ProductDependencies dependencies_;
    unsigned int files_ = 0;
    unsigned int count_ = 0;
  };
}  // namespace edmtest::pool

DEFINE_FWK_MODULE(edmtest::pool::TransitionValueProducer);
DEFINE_FWK_MODULE(edmtest::pool::TransitionValueChecker);
