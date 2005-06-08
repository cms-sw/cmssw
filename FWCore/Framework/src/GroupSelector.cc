// $Id: GroupSelector.cc,v 1.3 2005/06/03 04:02:22 wmtan Exp $
#include "FWCore/CoreFramework/interface/EventPrincipal.h"
#include "FWCore/CoreFramework/interface/EventProvenance.h"
#include "FWCore/CoreFramework/interface/GroupSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>

using namespace std;

namespace edm {
  GroupSelector::GroupSelector(ParameterSet const& pset) :
    selectAllGroups_(true),
    select_() {
    selectProducts(pset);
  }

  bool GroupSelector::selected(std::string const& label) const {
      SelectMap::const_iterator s = select_.find(label);
      bool const sel = selectAllGroups_ ?
	s == select_.end() || s->second : // select all branches, except those marked "select == false"
	s != select_.end() && s->second;  // select only branches marked "select == true"
      return sel;
  }

  void GroupSelector::selectProducts(ParameterSet const& pset) {
    bool selectOnlySpecified = pset.getBool("keepOnlySpecifiedProducts");
    if (selectOnlySpecified) {
      selectNone();
      vector<string> const keep = pset.getVString("keepProducts");
      for(vector<string>::const_iterator it = keep.begin(); it != keep.end(); ++it) {
        select(*it);
      }
    } else {
      selectAll();
      vector<string> const skip = pset.getVString("skipProducts");
      for(vector<string>::const_iterator it = skip.begin(); it != skip.end(); ++it) {
        unselect(*it);
      }
    }
  }

  void GroupSelector::selectAll() {
    selectAllGroups_ = true;
    select_.clear();
  }

  void GroupSelector::selectNone() {
    selectAllGroups_ = false;
    select_.clear();
  }

  void GroupSelector::select(std::string const& label) {
    select_[ label ] = true;
  }

  void GroupSelector::unselect(std::string const& label) {
    select_[ label ] = false;
  }
}
