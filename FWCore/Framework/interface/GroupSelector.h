#ifndef Framework_GroupSelector_h
#define Framework_GroupSelector_h

//////////////////////////////////////////////////////////////////////
//
// $Id: GroupSelector.h,v 1.6 2005/09/01 05:27:41 wmtan Exp $
//
// Class GroupSelector. Class for user to select specific groups in event.
//
// Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include <map>
#include <iosfwd>
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class ParameterSet;

  class GroupSelector {
  public:
    explicit GroupSelector(ParameterSet const& ps);
    ~GroupSelector() {}
    bool selected(BranchDescription const& desc) const;

  private:
    typedef std::map<std::string, bool> SelectMap;
    void selectProducts(ParameterSet const& pset);

  private:
    bool selectAllGroups_;
    SelectMap select_;
  };
}

#endif
