// $Id: GroupSelector.cc,v 1.9 2005/07/21 16:47:32 wmtan Exp $
#include "FWCore/Framework/interface/ProductDescription.h"
#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>

namespace edm {
  GroupSelector::GroupSelector(ParameterSet const& pset) :
    selectAllGroups_(false),
    select_() {
    selectProducts(pset);
  }

  bool GroupSelector::selected(ProductDescription const& desc) const {
      SelectMap::const_iterator s = select_.find(desc.module.module_label);
      bool const sel = selectAllGroups_ ?
	s == select_.end() || s->second : // select all branches, except those marked "select == false"
	s != select_.end() && s->second;  // select only branches marked "select == true"
      return sel;
  }

  void GroupSelector::selectProducts(ParameterSet const& pset) {
    std::string allString("*");
    std::vector<std::string> all;
    all.push_back(allString);
    std::vector<std::string> none;

    std::vector<std::string> keep = pset.getUntrackedParameter<std::vector<std::string> >("productsSelected", all);

    for(std::vector<std::string>::const_iterator it = keep.begin(); it != keep.end(); ++it) {
      std::string const& label = *it;
      if (allString == label) {
        selectAllGroups_ = true;
        select_.clear();
        break;
      } else {
        select_[label] = true;
      }
    }

    std::vector<std::string> skip = pset.getUntrackedParameter<std::vector<std::string> >("productsExcluded", none);

    for(std::vector<std::string>::const_iterator it = skip.begin(); it != skip.end(); ++it) {
      std::string const& label = *it;
      if (allString == label) {
        selectAllGroups_ = false;
        select_.clear();
        break;
      } else {
        select_[ label ] = false;
      }
    }
  }
}
