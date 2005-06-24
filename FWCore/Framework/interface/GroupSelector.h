#ifndef EDM_OUTPUTSELECTOR_H
#define EDM_OUTPUTSELECTOR_H

//////////////////////////////////////////////////////////////////////
//
// $Id: GroupSelector.h,v 1.2 2005/06/10 05:33:42 wmtan Exp $
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
    bool selected(Provenance const& prov) const;

  private:
    typedef std::map<std::string, bool> SelectMap;
    void selectProducts(ParameterSet const& pset);

  private:
    bool selectAllGroups_;
    SelectMap select_;
  };
}

#endif
