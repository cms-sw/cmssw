
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {
  void getAncestors(edm::Event const& e,
                    edm::BranchID const& branchID,
                    std::set<edm::BranchID>& ancestors) {
    edm::Provenance prov = e.getProvenance(branchID);
    for (auto const& parent : prov.productProvenance()->parentage().parents()) {
      ancestors.insert(parent);
      getAncestors(e, parent, ancestors);
    }
  }

  // Does the same thing as the previous function in a different
  // way. The previous function goes through the links in the
  // ProductsResolver which for SubProcesses could lead to a different
  // retriever. In SubProcesses, the following function follows the
  // links in the retrievers themselves. Both should give the same answer.
  void getAncestorsFromRetriever(edm::ProductProvenanceRetriever const* retriever,
                                 edm::BranchID const& branchID,
                                 std::set<edm::BranchID>& ancestors) {
    edm::ProductProvenance const* productProvenance = retriever->branchIDToProvenance(branchID);
    if (productProvenance) {
      for (auto const& parent : productProvenance->parentage().parents()) {
        ancestors.insert(parent);
        getAncestorsFromRetriever(retriever, parent, ancestors);
      }
    }
  }
}

namespace edmtest {

  class TestParentage : public edm::EDAnalyzer { 
  public:

    explicit TestParentage(edm::ParameterSet const& pset);
    virtual ~TestParentage();

    virtual void analyze(edm::Event const& e, edm::EventSetup const& es) override;

  private:

    edm::InputTag inputTag_;
    edm::EDGetTokenT<IntProduct> token_;
    std::vector<std::string> expectedAncestors_;
  };

  TestParentage::TestParentage(edm::ParameterSet const& pset) :
    inputTag_(pset.getParameter<edm::InputTag>("inputTag")),
    expectedAncestors_(pset.getParameter<std::vector<std::string> >("expectedAncestors")) {

    token_ = consumes<IntProduct>(inputTag_);
  }

  TestParentage::~TestParentage() {}

  void
  TestParentage::analyze(edm::Event const& e, edm::EventSetup const&) {

    edm::Handle<IntProduct> h;
    e.getByToken(token_, h);

    edm::Provenance const* prov = h.provenance();
    std::set<edm::BranchID> ancestors;
    getAncestors(e, prov->branchID(), ancestors);

    std::set<std::string> ancestorLabels;
    for (edm::BranchID const& ancestor : ancestors) {
      edm::Provenance ancestorProv = e.getProvenance(ancestor);
      ancestorLabels.insert(ancestorProv.moduleLabel());
    }
    std::set<std::string> expectedAncestors(expectedAncestors_.begin(), expectedAncestors_.end());
    if (ancestorLabels != expectedAncestors) {
      std::cerr << "TestParentage::analyze: ancestors do not match expected ancestors" << std::endl;
      abort();
    }
    edm::ProductProvenanceRetriever const* retriever = prov->store();
    std::set<edm::BranchID> ancestorsFromRetriever;
    getAncestorsFromRetriever(retriever, prov->branchID(), ancestorsFromRetriever);

    std::set<std::string> ancestorLabels2;
    for (edm::BranchID const& ancestor : ancestorsFromRetriever) {
      edm::Provenance ancestorProv = e.getProvenance(ancestor);
      ancestorLabels2.insert(ancestorProv.moduleLabel());
    }
    if (ancestorLabels2 != expectedAncestors) {
      std::cerr << "TestParentage::analyze: ancestors do not match expected ancestors (parentage from retriever)" << std::endl;
      abort();
    }
  }
} // namespace edmtest

using edmtest::TestParentage;
DEFINE_FWK_MODULE(TestParentage);
