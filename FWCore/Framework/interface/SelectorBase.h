#ifndef Framework_SelectorBase_h
#define Framework_SelectorBase_h

/*----------------------------------------------------------------------
  
Selector: Base class for all "selector" objects, used to select
EDProducts based on information in the associated Provenance.

Developers who make their own Selectors should inherit from SelectorBase.

----------------------------------------------------------------------*/
#include <typeinfo>
#include "FWCore/Utilities/interface/value_ptr.h"

namespace edm 
{
  class ConstBranchDescription;

  //------------------------------------------------------------------
  //
  //// Abstract base class SelectorBase
  //
  //------------------------------------------------------------------

  class SelectorBase {
  public:
    virtual ~SelectorBase();
    bool match(ConstBranchDescription const& p) const;
    bool matchSelectorType(std::type_info const& type) const;
    virtual SelectorBase* clone() const = 0;

  private:
    virtual bool doMatch(ConstBranchDescription const& p) const = 0;
    virtual bool doMatchSelectorType(std::type_info const& type) const = 0;
  };

  template <>
  struct value_ptr_traits<SelectorBase> {
    static SelectorBase* clone(SelectorBase const * p) { return p->clone(); }
  };
}

#endif
