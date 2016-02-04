

#include "EventFilter/StorageManager/test/SillyLockService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <algorithm>

using namespace edm;
using namespace std;

namespace Nuts {

  SillyLockService::SillyLockService(const ParameterSet& iPS, 
				     ActivityRegistry& reg):
    lock_(&edm::Service<stor::HLTInfo>()->getExtraLock()),
    locker_(),
    labels_(iPS.getUntrackedParameter<Labels>("labels",Labels()))
  {

    // reg.watchPostBeginJob(this,&SillyLockService::postBeginJob);
    // reg.watchPostEndJob(this,&SillyLockService::postEndJob);
    
    // reg.watchPreProcessEvent(this,&SillyLockService::preEventProcessing);
    // reg.watchPostProcessEvent(this,&SillyLockService::postEventProcessing);
    
    reg.watchPreModule(this,&SillyLockService::preModule);
    reg.watchPostModule(this,&SillyLockService::postModule);

    FDEBUG(4) << "In SillyLockServices" << std::endl;
  }


  SillyLockService::~SillyLockService()
  {
    delete locker_;
  }

  void SillyLockService::postBeginJob()
  {
  }

  void SillyLockService::postEndJob()
  {
  }

  void SillyLockService::preEventProcessing(const edm::EventID& iID,
				       const edm::Timestamp& iTime)
  {
  }

  void SillyLockService::postEventProcessing(const Event& e, const EventSetup&)
  {
  }

  void SillyLockService::preModule(const ModuleDescription& desc)
  {
    if(!labels_.empty() &&
       find(labels_.begin(),labels_.end(),desc.moduleLabel())!=labels_.end())
      {
	FDEBUG(4) << "made a new locked in SillyLockService" << std::endl;
	locker_ = new boost::mutex::scoped_lock(*lock_);
      }
  }

  void SillyLockService::postModule(const ModuleDescription& desc)
  {
    FDEBUG(4) << "destroyed a locked in SillyLockService" << std::endl;
    delete locker_;
    locker_=0;
  }

}

using Nuts::SillyLockService;
DEFINE_FWK_SERVICE(SillyLockService);

