#ifndef Framework_SelectorBase_h
#define Framework_SelectorBase_h

/*----------------------------------------------------------------------
  
Selector: Base class for all "selector" objects, used to select
EDProducts based on information in the associated Provenance.

Developers who make their own Selectors should inherit from SelectorBase.

$Id: SelectorBase.h,v 1.3.6.1 2008/05/12 15:33:08 wmtan Exp $

----------------------------------------------------------------------*/

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
    virtual SelectorBase* clone() const = 0;

  private:
    virtual bool doMatch(ConstBranchDescription const& p) const = 0;
  };
}

#endif
