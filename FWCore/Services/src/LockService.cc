#include "FWCore/Services/src/LockService.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/GlobalMutex.h"

#include <algorithm>
#include <iostream>

using namespace edm::rootfix;

LockService::LockService(ParameterSet const& iPS,
                         ActivityRegistry& reg):
  lock_(*getGlobalMutex()),
  locker_(),
  labels_(iPS.getUntrackedParameter<Labels>("labels")),
  lockSources_(iPS.getUntrackedParameter<bool>("lockSources")) {
  reg.watchPreSourceConstruction(this,&LockService::preSourceConstruction);
  reg.watchPostSourceConstruction(this,&LockService::postSourceConstruction);

  // reg.watchPostBeginJob(this,&LockService::postBeginJob);
  // reg.watchPostEndJob(this,&LockService::postEndJob);

  // reg.watchPreProcessEvent(this,&LockService::preEventProcessing);
  // reg.watchPostProcessEvent(this,&LockService::postEventProcessing);
  reg.watchPreSource(this,&LockService::preSource);
  reg.watchPostSource(this,&LockService::postSource);

  reg.watchPreModule(this,&LockService::preModule);
  reg.watchPostModule(this,&LockService::postModule);

  FDEBUG(4) << "In LockServices" << std::endl;
}

LockService::~LockService() {
}

void LockService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> defaultVector;
  desc.addUntracked<std::vector<std::string> >("labels",defaultVector);
  desc.addUntracked<bool>("lockSources",true);
  descriptions.add("LockService", desc);
}

void LockService::preSourceConstruction(ModuleDescription const& desc) {
  if(!labels_.empty() &&
     find(labels_.begin(), labels_.end(), desc.moduleLabel()) != labels_.end()) {
     //search_all(labels_, desc.moduleLabel()))
      FDEBUG(4) << "made a new locked in LockService" << std::endl;
      locker_.reset(new boost::mutex::scoped_lock(lock_));
    }
}

void LockService::postSourceConstruction(ModuleDescription const&) {
  FDEBUG(4) << "destroyed a locked in LockService" << std::endl;
  locker_.reset();
}

void LockService::postBeginJob() {
}

void LockService::postEndJob() {
}

void LockService::preEventProcessing(edm::EventID const&, edm::Timestamp const&) {
}

void LockService::postEventProcessing(Event const&, EventSetup const&) {
}

void LockService::preSource() {
  if(lockSources_) {
    FDEBUG(4) << "made a new locked in LockService" << std::endl;
    locker_.reset(new boost::mutex::scoped_lock(lock_));
  }
}

void LockService::postSource() {
  FDEBUG(4) << "destroyed a locked in LockService" << std::endl;
  locker_.reset();
}

void LockService::preModule(ModuleDescription const& desc) {
  if(!labels_.empty() && find(labels_.begin(), labels_.end(), desc.moduleLabel()) != labels_.end()) {
    //search_all(labels_, desc.moduleLabel()))
    FDEBUG(4) << "made a new locked in LockService" << std::endl;
    locker_.reset(new boost::mutex::scoped_lock(lock_));
  }
}

void LockService::postModule(ModuleDescription const&) {
  FDEBUG(4) << "destroyed a locked in LockService" << std::endl;
  locker_.reset();
}

