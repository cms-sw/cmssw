#ifndef Framework_InputSourceDescription_h
#define Framework_InputSourceDescription_h

/*----------------------------------------------------------------------

InputSourceDescription : the stuff that is needed to configure an
input source that does not come in through the ParameterSet  

$Id: InputSourceDescription.h,v 1.3.2.2 2006/07/05 23:57:17 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/ProcessConfiguration.h"

namespace edm {
  class ProductRegistry;

  struct InputSourceDescription {
    InputSourceDescription() : moduleDescription_(), productRegistry_(0) { }
    InputSourceDescription(ModuleDescription const& md,
			    ProductRegistry& preg) :
      moduleDescription_(md),
      productRegistry_(&preg)
	 
    {}

    ModuleDescription moduleDescription_;
    ProductRegistry * productRegistry_;
  };
}

#endif
