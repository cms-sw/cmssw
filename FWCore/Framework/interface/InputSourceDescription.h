#ifndef Framework_InputSourceDescription_h
#define Framework_InputSourceDescription_h

/*----------------------------------------------------------------------

InputSourceDescription : the stuff that is needed to configure an
input source that does not come in through the ParameterSet  

$Id: InputSourceDescription.h,v 1.5 2007/03/04 06:00:22 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"

namespace edm {
  class ProductRegistry;

  struct InputSourceDescription {
    InputSourceDescription() : moduleDescription_(), productRegistry_(0), maxEvents_(-1) {}
    InputSourceDescription(ModuleDescription const& md,
			   ProductRegistry& preg,
			   int maxEvents) :
      moduleDescription_(md),
      productRegistry_(&preg),
      maxEvents_(maxEvents)
	 
    {}

    ModuleDescription moduleDescription_;
    ProductRegistry * productRegistry_;
    int maxEvents_;
  };
}

#endif
