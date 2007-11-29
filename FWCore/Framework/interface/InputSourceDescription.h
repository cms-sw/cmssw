#ifndef FWCore_Framework_InputSourceDescription_h
#define FWCore_Framework_InputSourceDescription_h

/*----------------------------------------------------------------------

InputSourceDescription : the stuff that is needed to configure an
input source that does not come in through the ParameterSet  

$Id: InputSourceDescription.h,v 1.7 2007/06/14 17:52:15 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>
#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace edm {
  class ProductRegistry;

  struct InputSourceDescription {
    InputSourceDescription() : moduleDescription_(), productRegistry_(0), maxEvents_(-1), maxLumis_(-1) {}
    InputSourceDescription(ModuleDescription const& md,
			   ProductRegistry& preg,
			   int maxEvents,
			   int maxLumis) :
      moduleDescription_(md),
      productRegistry_(&preg),
      maxEvents_(maxEvents),
      maxLumis_(maxLumis)
	 
    {}

    ModuleDescription moduleDescription_;
    ProductRegistry * productRegistry_;
    int maxEvents_;
    int maxLumis_;
  };
}

#endif
