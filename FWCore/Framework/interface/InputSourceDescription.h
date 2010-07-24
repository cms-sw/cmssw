#ifndef FWCore_Framework_InputSourceDescription_h
#define FWCore_Framework_InputSourceDescription_h

/*----------------------------------------------------------------------

InputSourceDescription : the stuff that is needed to configure an
input source that does not come in through the ParameterSet  
----------------------------------------------------------------------*/
#include "boost/shared_ptr.hpp"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace edm {
  class PrincipalCache;
  class ProductRegistry;
  class ActivityRegistry;

  struct InputSourceDescription {
    InputSourceDescription() :
      moduleDescription_(),
      productRegistry_(0),
      principalCache_(0),
      actReg_(), maxEvents_(-1),
      maxLumis_(-1) {}
    InputSourceDescription(ModuleDescription const& md,
			   ProductRegistry& preg,
			   PrincipalCache& pCache,
			   boost::shared_ptr<ActivityRegistry> areg,
			   int maxEvents,
			   int maxLumis) :
      moduleDescription_(md),
      productRegistry_(&preg),
      principalCache_(&pCache),
      actReg_(areg),
      maxEvents_(maxEvents),
      maxLumis_(maxLumis)
	 
    {}

    ModuleDescription moduleDescription_;
    ProductRegistry* productRegistry_;
    PrincipalCache* principalCache_;
    boost::shared_ptr<ActivityRegistry> actReg_;
    int maxEvents_;
    int maxLumis_;
  };
}

#endif
