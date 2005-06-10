#ifndef EDM_OUTPUTSELECTOR_H
#define EDM_OUTPUTSELECTOR_H

//////////////////////////////////////////////////////////////////////
//
// $Id: GroupSelector.h,v 1.1 2005/06/08 21:15:55 wmtan Exp $
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
#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"

namespace edm {
  class ParameterSet;

  class GroupSelector {
  public:
    explicit GroupSelector(ParameterSet const& ps);
    ~GroupSelector() {}
    bool selected(std::string const& label) const;

  private:
    typedef std::map<std::string, bool> SelectMap;
    void selectProducts(ParameterSet const& pset);

  private:
    bool selectAllGroups_;
    SelectMap select_;
  };
}

#endif
