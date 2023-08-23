#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class EventSetup;
  class StreamID;
}  // namespace edm

namespace edmtest {
  class SwitchProducerProvenanceAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit SwitchProducerProvenanceAnalyzer(edm::ParameterSet const& iConfig);
    void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override;

  private:
    void testProduct(edm::Handle<IntProduct> const& prod, int mode, edm::Event const& iEvent) const;

    edm::EDGetTokenT<IntProduct> inputToken1_;
    edm::EDGetTokenT<IntProduct> inputToken2_;
    std::string producerPrefix_;
    bool const aliasMode_;
  };

  SwitchProducerProvenanceAnalyzer::SwitchProducerProvenanceAnalyzer(edm::ParameterSet const& iConfig)
      : inputToken1_(consumes(iConfig.getParameter<edm::InputTag>("src1"))),
        inputToken2_(consumes(iConfig.getParameter<edm::InputTag>("src2"))),
        producerPrefix_(iConfig.getParameter<std::string>("producerPrefix")),
        aliasMode_(iConfig.getParameter<bool>("aliasMode")) {}

  void SwitchProducerProvenanceAnalyzer::analyze(edm::StreamID,
                                                 edm::Event const& iEvent,
                                                 edm::EventSetup const& iSetup) const {
    edm::Handle<IntProduct> h;
    iEvent.getByToken(inputToken1_, h);
    testProduct(h, iEvent.id().luminosityBlock(), iEvent);

    iEvent.getByToken(inputToken2_, h);
    testProduct(h, iEvent.id().luminosityBlock(), iEvent);
  }

  void SwitchProducerProvenanceAnalyzer::testProduct(edm::Handle<IntProduct> const& prod,
                                                     int mode,
                                                     edm::Event const& iEvent) const {
    assert(prod->value == mode);

    edm::Provenance const* provenance = prod.provenance();
    assert(provenance != nullptr);
    auto const* productProvenance = provenance->productProvenance();
    assert(productProvenance != nullptr);
    auto const& processHistory = iEvent.processHistory();

    edm::pset::Registry const* psetRegistry = edm::pset::Registry::instance();
    assert(psetRegistry != nullptr);

    auto const& moduleLabel = provenance->moduleLabel();

    // Switch output should not look like an alias
    assert(productProvenance->branchID() == provenance->branchID());

    // Check that the provenance of the Switch itself is recorded correctly
    for (edm::ProcessConfiguration const& pc : processHistory) {
      if (pc.processName() == provenance->processName()) {
        edm::ParameterSetID const& psetID = pc.parameterSetID();
        edm::ParameterSet const* processPSet = psetRegistry->getMapped(psetID);
        assert(processPSet);
        auto const& modPSet = processPSet->getParameterSet(moduleLabel);
        assert(modPSet.getParameter<std::string>("@module_edm_type") == "EDProducer");
        assert(modPSet.getParameter<std::string>("@module_type") == "SwitchProducer");
        assert(modPSet.getParameter<std::string>("@module_label") == moduleLabel);
        auto const& allCases = modPSet.getParameter<std::vector<std::string>>("@all_cases");
        assert(allCases.size() == 2);
        assert(allCases[0] == moduleLabel + "@test1");
        assert(allCases[1] == moduleLabel + "@test2");
        assert(modPSet.exists("@chosen_case") == false);
      }
    }

    // Check the parentage (foo -> foo@case -> possible input)
    auto const& parent = productProvenance->parentage();
    // Here is where Switch differs from a normal EDProducer: each Switch output branch has exactly one parent
    assert(parent.parents().size() == 1);
    auto const& parentProvenance = iEvent.getProvenance(parent.parents()[0]);
    edm::ProductProvenance const* parentProductProvenance = nullptr;
    if (not(aliasMode_ and mode == 2)) {
      // If parent is EDAlias, it is skipped in the provenance, so in
      // that case the normal grandparent can be found on the place of
      // the parent.

      assert(parentProvenance.branchDescription().moduleLabel() == moduleLabel + "@test" + std::to_string(mode));

      // Check grandparent as well
      parentProductProvenance = parentProvenance.productProvenance();
      assert(parentProductProvenance != nullptr);
    }
    auto const& grandParent = parentProductProvenance ? parentProductProvenance->parentage() : parent;
    //auto const& grandParent = parentProductProvenance->parentage();
    assert(grandParent.parents().size() == 1);  // behaviour of the AddIntsProducer
    auto const& grandParentProvenance = iEvent.getProvenance(grandParent.parents()[0]);
    int postfix = mode;
    if (aliasMode_ and mode == 2) {
      postfix = 3;
    }
    assert(grandParentProvenance.branchDescription().moduleLabel() == producerPrefix_ + std::to_string(postfix));
  }
}  // namespace edmtest
using edmtest::SwitchProducerProvenanceAnalyzer;
DEFINE_FWK_MODULE(SwitchProducerProvenanceAnalyzer);
