#ifndef Framework_GeneratedInputSource_h
#define Framework_GeneratedInputSource_h

/*----------------------------------------------------------------------
$Id: GeneratedInputSource.h,v 1.1 2005/12/28 00:30:09 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/ConfigurableInputSource.h"

namespace edm {
  class GeneratedInputSource : public ConfigurableInputSource {
  public:
    explicit GeneratedInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~GeneratedInputSource();

  };
}
#endif
