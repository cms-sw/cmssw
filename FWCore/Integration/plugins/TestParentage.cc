
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProductProvenanceLookup.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {
  void getAncestors(edm::Event const& e, edm::BranchID const& branchID, std::set<edm::BranchID>& ancestors) {
    const edm::Provenance& prov = e.getProvenance(branchID);
    if (prov.productProvenance()) {
      for (auto const& parent : prov.productProvenance()->parentage().parents()) {
        ancestors.insert(parent);
        getAncestors(e, parent, ancestors);
      }
    }
  }

  // Does the same thing as the previous function in a different
  // way. The previous function goes through the links in the
  // ProductsResolver which for SubProcesses could lead to a different
  // retriever. In SubProcesses, the following function follows the
  // links in the retrievers themselves. Both should give the same answer.
  void getAncestorsFromRetriever(edm::ProductProvenanceLookup const* retriever,
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
}  // namespace

namespace edmtest {

  class TestParentage : public edm::global::EDAnalyzer<> {
  public:
    explicit TestParentage(edm::ParameterSet const& pset);
    ~TestParentage() override = default;

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& es) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& iDesc);

  private:
    edm::EDGetTokenT<IntProduct> token_;
    std::vector<std::string> expectedAncestors_;
    bool callGetProvenance_;
  };

  TestParentage::TestParentage(edm::ParameterSet const& pset)
      : token_(consumes(pset.getParameter<edm::InputTag>("inputTag"))),
        expectedAncestors_(pset.getParameter<std::vector<std::string> >("expectedAncestors")),
        callGetProvenance_(pset.getUntrackedParameter<bool>("callGetProvenance", true)) {}

  void TestParentage::fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription ps;
    ps.add<edm::InputTag>("inputTag");
    ps.add<std::vector<std::string> >("expectedAncestors")
        ->setComment(
            "Module labels for data products directly/indirectly obtained to make data product retrieved using "
            "'inputTag'.");
    ps.addUntracked<bool>("callGetProvenance", true)
        ->setComment("Use Event::getProvenance to get ancestor provenance.");

    iDesc.addDefault(ps);
  }

  void TestParentage::analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const&) const {
    edm::Handle<IntProduct> h = e.getHandle(token_);
    *h;
    edm::Provenance const* prov = h.provenance();

    if (not prov) {
      throw cms::Exception("MissingProvenance") << "Failed to get provenance for 'inputTag'";
    }
    if (prov->originalBranchID() != prov->branchDescription().originalBranchID()) {
      throw cms::Exception("InconsistentBranchID")
          << " test of Provenance::originalBranchID function failed. Expected "
          << prov->branchDescription().originalBranchID() << " but see " << prov->originalBranchID();
    }

    std::set<std::string> expectedAncestors(expectedAncestors_.begin(), expectedAncestors_.end());

    // Currently we need to turn off this part of the test of when calling
    // from a SubProcess and the parentage includes a product not kept
    // in the SubProcess. This might get fixed someday ...
    auto toException = [](auto& ex, auto const& ancestors) {
      for (auto const& a : ancestors) {
        ex << a << ", ";
      }
    };

    if (callGetProvenance_) {
      std::set<edm::BranchID> ancestors;
      getAncestors(e, prov->branchID(), ancestors);

      std::set<std::string> ancestorLabels;
      for (edm::BranchID const& ancestor : ancestors) {
        try {
          ancestorLabels.insert(e.getStableProvenance(ancestor).moduleLabel());
        } catch(cms::Exception& iEx) {
          edm::LogSystem("MissingProvenance") << "the provenance for BranchID " << ancestor << " is missing\n"<<iEx.what();
          ancestorLabels.insert("");
        }
      }
      if (ancestorLabels != expectedAncestors) {
        cms::Exception ex("WrongAncestors");
        ex << "ancestors from Event::getProvenance\n";
        toException(ex, ancestorLabels);
        ex << "\n do not match expected ancestors\n";
        toException(ex, expectedAncestors);
        throw ex;
      }
    }

    auto const* retriever = prov->store();
    std::set<edm::BranchID> ancestorsFromRetriever;
    getAncestorsFromRetriever(retriever, prov->originalBranchID(), ancestorsFromRetriever);

    std::set<std::string> ancestorLabels2;
    for (edm::BranchID const& ancestor : ancestorsFromRetriever) {
      try {
        ancestorLabels2.insert(e.getStableProvenance(ancestor).moduleLabel());
      } catch(cms::Exception& iEx) {
          edm::LogSystem("MissingProvenance") << "the provenance from Retriever for BranchID " << ancestor << " is missing\n"<<iEx.what();
          ancestorLabels2.insert("");
      }
    }
    if (ancestorLabels2 != expectedAncestors) {
      cms::Exception ex("WrongAncestors");
      ex << "ancestors from ParentageRetriever\n";
      toException(ex, ancestorLabels2);
      ex << "\n do not match expected ancestors\n";
      toException(ex, expectedAncestors);
      throw ex;
    }
  }
}  // namespace edmtest

using edmtest::TestParentage;
DEFINE_FWK_MODULE(TestParentage);
