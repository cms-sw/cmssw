#include "DataFormats/Provenance/interface/SubProcessParentageHelper.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/BranchType.h"

namespace edm {

  void SubProcessParentageHelper::update(SubProcessParentageHelper const& parentSubProcessParentageHelper,
                                         ProductRegistry const& parentProductRegistry) {
    *this = parentSubProcessParentageHelper;

    for (auto const& prod : parentProductRegistry.productList()) {
      BranchDescription const& desc = prod.second;
      if (desc.produced() && desc.branchType() == InEvent && !desc.isAlias()) {
        producedProducts_.push_back(desc.branchID());
      }
    }
  }
}  // namespace edm
