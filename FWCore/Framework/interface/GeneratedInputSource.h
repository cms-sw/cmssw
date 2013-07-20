#ifndef Framework_GeneratedInputSource_h
#define Framework_GeneratedInputSource_h

/*----------------------------------------------------------------------
$Id: GeneratedInputSource.h,v 1.2 2006/01/18 00:38:44 wmtan Exp $
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
