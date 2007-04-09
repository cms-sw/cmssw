#ifndef Framework_SelectorBase_h
#define Framework_SelectorBase_h

/*----------------------------------------------------------------------
  
Selector: Base class for all "selector" objects, used to select
EDProducts based on information in the associated Provenance.

Developers who make their own Selectors should inherit from SelectorBase.

$Id: SelectorBase.h,v 1.2 2007/03/04 06:00:22 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm 
{
  class SelectorProvenance;
  struct Provenance;

  //------------------------------------------------------------------
  //
  //// Abstract base class SelectorBase
  //
  //------------------------------------------------------------------

  class SelectorBase {
  public:
    virtual ~SelectorBase();
    bool match(Provenance const& p) const;
    bool match(SelectorProvenance const& p) const;
    virtual SelectorBase* clone() const = 0;

  private:
    virtual bool doMatch(SelectorProvenance const& p) const = 0;
  };
}

#endif
