#ifndef Framework_InputSourceDescription_h
#define Framework_InputSourceDescription_h

/*----------------------------------------------------------------------

InputSourceDescription : the stuff that is needed to configure an
input source that does not come in through the ParameterSet  

$Id: InputSourceDescription.h,v 1.2 2006/02/08 00:44:24 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>
#include "DataFormats/Common/interface/ModuleDescription.h"

namespace edm {
  class ProductRegistry;

  struct InputSourceDescription {
    InputSourceDescription() : module_(), preg_(0) { }
    InputSourceDescription(ModuleDescription & md,
			    ProductRegistry& preg) :
      module_(md),
      preg_(&preg)
	 
    {}

    ModuleDescription module_;
    ProductRegistry * preg_;
  };
}

#endif
