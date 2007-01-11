#include "IOPool/Streamer/interface/StreamerInputSource.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <iostream>

namespace edm {

  StreamerInputSource::StreamerInputSource(
                    ParameterSet const& pset,
                    InputSourceDescription const& desc):
    InputSource(pset, desc),
    deserializer_()
  {
  }

  StreamerInputSource::~StreamerInputSource() {}

  void
  StreamerInputSource::mergeWithRegistry(SendDescs const& descs,ProductRegistry& reg) {

    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct

    FDEBUG(6) << "mergeWithRegistry: Product List: " << std::endl;
    std::string processName;
    if( i != e ) {
       processName = i->processName();
    }
    for(; i != e; ++i) {
        reg.copyProduct(*i);
        FDEBUG(6) << "StreamInput prod = " << i->className() << std::endl;

	if(processName != i->processName()) {
	   throw cms::Exception("MultipleProcessNames")
	      <<"at least two different process names ('"
	      <<processName
	      <<"', '"
	      <<i->processName()
	      <<"' found in JobHeader. We can only support one.";
	}
    }
    deserializer_.setProcessConfiguration(
	ProcessConfiguration(processName, ParameterSetID(), ReleaseVersion(), PassID()));

  }

  void
  StreamerInputSource::declareStreamers(SendDescs const& descs) {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
        //pi->init();
        std::string const real_name = wrappedClassName(i->className());
        FDEBUG(6) << "declare: " << real_name << std::endl;
        loadCap(real_name);
    }
  }


  void
  StreamerInputSource::buildClassCache(SendDescs const& descs) { 
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
        //pi->init();
        std::string const real_name = wrappedClassName(i->className());
        FDEBUG(6) << "BuildReadData: " << real_name << std::endl;
        doBuildRealData(real_name);
    }
  }

} // end of namespace-edm
