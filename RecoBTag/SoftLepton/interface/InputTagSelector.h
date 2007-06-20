#ifndef edm_InputTagSelector_h
#define edm_InputTagSelector_h

#include "FWCore/Framework/interface/SelectorBase.h"
#include "FWCore/Framework/interface/SelectorProvenance.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm {

class InputTagSelector : public SelectorBase
{
public:
  explicit
  InputTagSelector(edm::InputTag const& tag) : tag_(tag)
  {
  }

  virtual bool doMatch(SelectorProvenance const& p) const
  {
    return (
        (p.moduleLabel() == tag.label()) and
        (p.productInstanceName() == tag.instance()) and
        (tag.process().empty() or tag.process() == "*" or p.processName() == tag.process())
    );
  }

  virtual InputTagSelector* clone(void) const {
    return new InputTagSelector(*this);
  }

private:
  edm::InputTag tag_;
};

#endif // edm_InputTagSelector_h
