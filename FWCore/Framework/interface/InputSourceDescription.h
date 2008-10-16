#ifndef FWCore_Framework_InputSourceDescription_h
#define FWCore_Framework_InputSourceDescription_h

/*----------------------------------------------------------------------

InputSourceDescription : the stuff that is needed to configure an
input source that does not come in through the ParameterSet  

$Id: InputSourceDescription.h,v 1.8 2007/11/29 17:27:38 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>
#include "boost/shared_ptr.hpp"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace edm {
  class ProductRegistry;
  class ActivityRegistry;

  struct InputSourceDescription {
    InputSourceDescription() : moduleDescription_(), productRegistry_(0), actReg_(), maxEvents_(-1), maxLumis_(-1) {}
    InputSourceDescription(ModuleDescription const& md,
			   ProductRegistry& preg,
			   boost::shared_ptr<ActivityRegistry> areg,
			   int maxEvents,
			   int maxLumis) :
      moduleDescription_(md),
      productRegistry_(&preg),
      actReg_(areg),
      maxEvents_(maxEvents),
      maxLumis_(maxLumis)
	 
    {}

    ModuleDescription moduleDescription_;
    ProductRegistry * productRegistry_;
    boost::shared_ptr<ActivityRegistry> actReg_;
    int maxEvents_;
    int maxLumis_;
  };
}

#endif
