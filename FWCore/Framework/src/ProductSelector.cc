#include <algorithm>
#include <iterator>
#include <ostream>
#include <cctype>

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"

namespace edm {
  // The following typedef is used only in this implementation file, in
  // order to shorten several lines of code.
  typedef std::vector<edm::BranchDescription const*> VCBDP;

  ProductSelector::ProductSelector() : productsToSelect_(), initialized_(false) {}

  void ProductSelector::initialize(ProductSelectorRules const& rules, VCBDP const& branchDescriptions) {
    typedef ProductSelectorRules::BranchSelectState BranchSelectState;

    // Get a BranchSelectState for each branch, containing the branch
    // name, with its 'select bit' set to false.
    std::vector<BranchSelectState> branchstates;
    {
      branchstates.reserve(branchDescriptions.size());

      VCBDP::const_iterator it = branchDescriptions.begin();
      VCBDP::const_iterator end = branchDescriptions.end();
      for (; it != end; ++it)
        branchstates.emplace_back(*it);
    }

    // Now  apply the rules to  the branchstates, in order.  Each rule
    // can override any previous rule, or all previous rules.
    rules.applyToAll(branchstates);

    // For each of the BranchSelectStates that indicates the branch is
    // to be selected, remember the branch name.  The list of branch
    // names must be sorted, for the implementation of 'selected' to
    // work.
    {
      std::vector<BranchSelectState>::const_iterator it = branchstates.begin();
      std::vector<BranchSelectState>::const_iterator end = branchstates.end();
      for (; it != end; ++it) {
        if (it->selectMe)
          productsToSelect_.push_back(it->desc->branchName());
      }
      sort_all(productsToSelect_);
    }
    initialized_ = true;
  }

  bool ProductSelector::selected(BranchDescription const& desc) const {
    if (!initialized_) {
      throw edm::Exception(edm::errors::LogicError) << "ProductSelector::selected() called prematurely\n"
                                                    << "before the product registry has been frozen.\n";
    }
    // We are to select this 'branch' if its name is one of the ones we
    // have been told to select.
    return binary_search_all(productsToSelect_, desc.branchName());
  }

  void ProductSelector::print(std::ostream& os) const {
    os << "ProductSelector at: " << static_cast<void const*>(this) << " has " << productsToSelect_.size()
       << " products to select:\n";
    copy_all(productsToSelect_, std::ostream_iterator<std::string>(os, "\n"));
  }

  void ProductSelector::checkForDuplicateKeptBranch(
      BranchDescription const& desc, std::map<BranchID, BranchDescription const*>& trueBranchIDToKeptBranchDesc) {
    // Check if an equivalent branch has already been selected due to an EDAlias.
    // We only need the check for products produced in this process.
    if (desc.produced()) {
      auto check = [&](BranchID const& branchID) {
        auto iter = trueBranchIDToKeptBranchDesc.find(branchID);
        if (iter != trueBranchIDToKeptBranchDesc.end()) {
          throw edm::Exception(errors::Configuration, "Duplicate Output Selection")
              << "Two (or more) equivalent branches have been selected for output.\n"
              << "#1: " << BranchKey(desc) << "\n"
              << "#2: " << BranchKey(*iter->second) << "\n"
              << "Please drop at least one of them.\n";
        }
      };
      // In case of SwitchProducer, we have to check the aliased-for
      // BranchID for the case that the chosen case is an EDAlias
      BranchID const& trueBranchID = desc.isSwitchAlias() ? desc.switchAliasForBranchID() : desc.originalBranchID();
      check(trueBranchID);
      trueBranchIDToKeptBranchDesc.insert(std::make_pair(trueBranchID, &desc));
    }
  }

  // Fills in a mapping needed in the case that a branch was dropped while its EDAlias was kept.
  void ProductSelector::fillDroppedToKept(
      ProductRegistry const& preg,
      std::map<BranchID, BranchDescription const*> const& trueBranchIDToKeptBranchDesc,
      std::map<BranchID::value_type, BranchID::value_type>& droppedBranchIDToKeptBranchID_) {
    for (auto const& it : preg.productList()) {
      BranchDescription const& desc = it.second;
      if (!desc.produced() || desc.isAlias())
        continue;
      BranchID const& branchID = desc.branchID();
      std::map<BranchID, BranchDescription const*>::const_iterator iter = trueBranchIDToKeptBranchDesc.find(branchID);
      if (iter != trueBranchIDToKeptBranchDesc.end()) {
        // This branch, produced in this process, or an alias of it, was persisted.
        BranchID const& keptBranchID = iter->second->branchID();
        if (keptBranchID != branchID) {
          // An EDAlias branch was persisted.
          droppedBranchIDToKeptBranchID_.insert(std::make_pair(branchID.id(), keptBranchID.id()));
        }
      }
    }
  }

  //--------------------------------------------------
  //
  // Associated free functions
  //
  std::ostream& operator<<(std::ostream& os, const ProductSelector& gs) {
    gs.print(os);
    return os;
  }

}  // namespace edm
