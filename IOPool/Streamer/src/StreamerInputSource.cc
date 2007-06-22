#include "IOPool/Streamer/interface/StreamerInputSource.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <vector>
#include <string>
#include <iostream>

namespace edm {

  StreamerInputSource::StreamerInputSource(
                    ParameterSet const& pset,
                    InputSourceDescription const& desc):
    InputSource(pset, desc),
    deserializer_(),
    holder_(0)
  {
  }

  StreamerInputSource::~StreamerInputSource() {}

  // ---------------------------------------

  void
  StreamerInputSource::mergeIntoRegistry(SendDescs const& descs,
			 ProductRegistry& reg) {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct

    FDEBUG(6) << "mergeIntoRegistry: Product List: " << std::endl;
    for(; i != e; ++i) {
	reg.copyProduct(*i);
	FDEBUG(6) << "StreamInput prod = " << i->className() << std::endl;
    }

    // not needed any more
    // fillStreamers(*pr_);
  }

  void
  StreamerInputSource::mergeWithRegistry(SendDescs const& descs, ProductRegistry& reg) {

    mergeIntoRegistry(descs, reg);

    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct

    std::string processName;

    //process name is already stored in deserializer_, if for Protocol version 4 and above.
    //From the INIT Message itself.
    
    if ( deserializer_.protocolVersion_ > 3) {
           processName = deserializer_.processName_;
    } else { 
    	if (i != e) {
       		processName = i->processName();
	}
    	for (; i != e; ++i) {
		if(processName != i->processName()) {
	   	throw cms::Exception("MultipleProcessNames")
	      		<< "at least two different process names ('"
	      		<< processName
	      		<< "', '"
	      		<< i->processName()
	      		<< "' found in JobHeader. We can only support one.";
		}
    	}
    }

    FDEBUG(10) << "StreamerInputSource::mergeWithRegistry :"<<processName<<std::endl; 

    edm::ProcessHistory ph;
    ph.reserve(1);
    ph.push_back(edm::ProcessConfiguration(processName, ParameterSetID(), ReleaseVersion(), PassID()));
    edm::ProcessHistoryRegistry::instance()->insertMapped(ph);
    deserializer_.setProcessConfiguration(processConfiguration());
    deserializer_.setProcessHistoryID(ph.id());

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

  void 
  StreamerInputSource::saveTriggerNames(InitMsgView const* header) {

    ParameterSet trigger_pset;
    std::vector<std::string> paths;
    header->hltTriggerNames(paths);
    trigger_pset.addParameter<Strings>("@trigger_paths", paths);
    pset::Registry* psetRegistry = pset::Registry::instance();
    psetRegistry->insertMapped(trigger_pset);
  }

  boost::shared_ptr<RunPrincipal>
  StreamerInputSource::readRun_() {
    if (holder_.get() == 0) holder_ = read();
    if (holder_.get() == 0) return boost::shared_ptr<RunPrincipal>();
    return boost::shared_ptr<RunPrincipal>(
	new RunPrincipal(holder_->runNumber(),
			 productRegistry(),
			 holder_->processConfiguration()));
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  StreamerInputSource::readLuminosityBlock_(boost::shared_ptr<RunPrincipal> rp) {
    if (holder_.get() == 0) holder_ = read();
    if (holder_.get() == 0 || rp->run() != holder_->runNumber()) {
      return boost::shared_ptr<LuminosityBlockPrincipal>();
    }
    return boost::shared_ptr<LuminosityBlockPrincipal>(
	new LuminosityBlockPrincipal(holder_->luminosityBlock(),
				     productRegistry(),
				     rp,
				     holder_->processConfiguration()));
  }

  std::auto_ptr<EventPrincipal>
  StreamerInputSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    if (holder_.get() == 0) holder_ = read();
    if (holder_.get() == 0 ||
        lbp->runNumber() != holder_->runNumber() ||
	lbp->luminosityBlock() != holder_->luminosityBlock()) {
      return std::auto_ptr<EventPrincipal>(0);
    }
    return holder_;
  }

} // end of namespace-edm
