#ifndef EDM_OUTPUTSELECTOR_H
#define EDM_OUTPUTSELECTOR_H

//////////////////////////////////////////////////////////////////////
//
// $Id: GroupSelector.h,v 1.3 2005/06/24 23:32:13 wmtan Exp $
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
