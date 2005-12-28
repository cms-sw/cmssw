#ifndef Framework_GroupSelector_h
#define Framework_GroupSelector_h

//////////////////////////////////////////////////////////////////////
//
// $Id: GroupSelector.h,v 1.7 2005/10/03 19:03:40 wmtan Exp $
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

namespace edm {
  class BranchDescription;
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
