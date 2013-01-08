#include <algorithm>
#include <iterator>
#include <ostream>
#include <cctype>

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"


namespace edm {
// The following typedef is used only in this implementation file, in
// order to shorten several lines of code.
  typedef std::vector<edm::BranchDescription const*> VCBDP;

  ProductSelector::ProductSelector() : productsToSelect_(), initialized_(false) {}

  void
  ProductSelector::initialize(ProductSelectorRules const& rules, VCBDP const& branchDescriptions) {
    typedef ProductSelectorRules::BranchSelectState BranchSelectState;

    // Get a BranchSelectState for each branch, containing the branch
    // name, with its 'select bit' set to false.
    std::vector<BranchSelectState> branchstates;
    {
      branchstates.reserve(branchDescriptions.size());
      
      VCBDP::const_iterator it = branchDescriptions.begin();
      VCBDP::const_iterator end = branchDescriptions.end();
      for (; it != end; ++it) branchstates.push_back(BranchSelectState(*it));
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
	  if (it->selectMe) productsToSelect_.push_back(it->desc->branchName());
      }
      sort_all(productsToSelect_);
    }
    initialized_ = true;
  }

  bool ProductSelector::selected(BranchDescription const& desc) const {
    if (!initialized_) {
      throw edm::Exception(edm::errors::LogicError)
        << "ProductSelector::selected() called prematurely\n"
        << "before the product registry has been frozen.\n";
    }
    // We are to select this 'branch' if its name is one of the ones we
    // have been told to select.
    return binary_search_all(productsToSelect_, desc.branchName());
  }

  void
  ProductSelector::print(std::ostream& os) const {
    os << "ProductSelector at: "
       << static_cast<void const*>(this)
       << " has "
       << productsToSelect_.size()
       << " products to select:\n";      
    copy_all(productsToSelect_, std::ostream_iterator<std::string>(os, "\n"));
  }


  //--------------------------------------------------
  //
  // Associated free functions
  //
  std::ostream&
  operator<< (std::ostream& os, const ProductSelector& gs)
  {
    gs.print(os);
    return os;
  }
  
}
