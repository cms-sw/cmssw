#include <algorithm>
#include <iterator>
#include <ostream>
#include <cctype>

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/Framework/interface/GroupSelectorRules.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"


namespace edm {
// The following typedef is used only in this implementation file, in
// order to shorten several lines of code.
  typedef std::vector<edm::BranchDescription const*> VCBDP;

  GroupSelector::GroupSelector() : groupsToSelect_(), initialized_(false) {}

  void
  GroupSelector::initialize(GroupSelectorRules const& rules, VCBDP const& branchDescriptions) {
    typedef GroupSelectorRules::BranchSelectState BranchSelectState;

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
	  if (it->selectMe) groupsToSelect_.push_back(it->desc->branchName());
      }
      sort_all(groupsToSelect_);
    }
    initialized_ = true;
  }

  bool GroupSelector::selected(BranchDescription const& desc) const {
    if (!initialized_) {
      throw edm::Exception(edm::errors::LogicError)
        << "GroupSelector::selected() called prematurely\n"
        << "before the product registry has been frozen.\n";
    }
    // We are to select this 'branch' if its name is one of the ones we
    // have been told to select.
    return binary_search_all(groupsToSelect_, desc.branchName());
  }

  void
  GroupSelector::print(std::ostream& os) const {
    os << "GroupSelector at: "
       << static_cast<void const*>(this)
       << " has "
       << groupsToSelect_.size()
       << " groups to select:\n";      
    copy_all(groupsToSelect_, std::ostream_iterator<std::string>(os, "\n"));
  }


  //--------------------------------------------------
  //
  // Associated free functions
  //
  std::ostream&
  operator<< (std::ostream& os, const GroupSelector& gs)
  {
    gs.print(os);
    return os;
  }
  
}
