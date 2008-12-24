#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/BranchIDListRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <limits>

namespace edm {

  void
  BranchIDListHelper:: updateFromInput(BranchIDLists const& bidlists, std::string const& fileName) {
    typedef BranchIDListRegistry::const_iterator iter;
    BranchIDListRegistry& breg = *BranchIDListRegistry::instance();
    BranchIDListRegistry::collection_type& bdata = breg.data();
    iter j = bidlists.begin(), jEnd = bidlists.end();
    for(iter i = bdata.begin(), iEnd = bdata.end(); j != jEnd && i != iEnd; ++j, ++i) {
      if (*i != *j) {
	throw edm::Exception(errors::UnimplementedFeature)
	  << "Cannot merge file '" << fileName << "' due to a branch mismatch.\n"
	  << "Contact the framework group.\n";
      }
    }
    BranchIDToIndexMap& branchIDToIndexMap = breg.extra().branchIDToIndexMap_;
    for (; j != jEnd; ++j) {
      BranchListIndex blix = breg.data().size();
      breg.insertMapped(*j);
      for (BranchIDList::const_iterator i = j->begin(), iEnd = j->end(); i != iEnd; ++i) {
        ProductIndex pix = i - j->begin();
	branchIDToIndexMap.insert(std::make_pair(*i, std::make_pair(blix, pix)));
      }
    }
  }

  void
  BranchIDListHelper::updateRegistries(ProductRegistry const& preg) {
    BranchIDList bidlist;
    // Add entries for current process for ProductID to BranchID mapping.
    for (ProductRegistry::ProductList::const_iterator it = preg.productList().begin(), itEnd = preg.productList().end();
        it != itEnd; ++it) {
      if (it->second.produced()) {
        if (it->second.branchType() == InEvent) {
          bidlist.push_back(it->second.branchID().id());
        }
      }
    }
    BranchIDListRegistry& breg = *BranchIDListRegistry::instance();
    BranchIDToIndexMap& branchIDToIndexMap = breg.extra().branchIDToIndexMap_;
    if (!bidlist.empty()) {
      BranchListIndex blix = breg.data().size();
      breg.extra().producedBranchListIndex_ = blix;
      breg.insertMapped(bidlist);
      for (BranchIDList::const_iterator i = bidlist.begin(), iEnd = bidlist.end(); i != iEnd; ++i) {
        ProductIndex pix = i - bidlist.begin();
	branchIDToIndexMap.insert(std::make_pair(*i, std::make_pair(blix, pix)));
      }
    } else {
      breg.extra().producedBranchListIndex_ = std::numeric_limits<BranchListIndex>::max();
    }
  }

  void
  BranchIDListHelper::clearRegistries() {
    BranchIDListRegistry& breg = *BranchIDListRegistry::instance();
    breg.data().clear();
    breg.extra().branchIDToIndexMap_.clear();
  }
}
