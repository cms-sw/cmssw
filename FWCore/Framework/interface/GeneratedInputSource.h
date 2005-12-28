#ifndef Framework_GeneratedInputSource_h
#define Framework_GeneratedInputSource_h

/*----------------------------------------------------------------------
$Id: GeneratedInputSource.h,v 1.1 2005/10/17 19:22:41 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/GenericInputSource.h"

namespace edm {
  class GeneratedInputSource : public GenericInputSource {
  public:
    explicit GeneratedInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~GeneratedInputSource();

  };
}
#endif
