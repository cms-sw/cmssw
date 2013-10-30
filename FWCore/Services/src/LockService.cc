#include "FWCore/Services/src/LockService.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/GlobalMutex.h"

#include <algorithm>
#include <iostream>

using namespace edm::rootfix;

LockService::LockService(ParameterSet const& iPS,
                         ActivityRegistry& reg):
  lock_(*getGlobalMutex()),
  labels_(iPS.getUntrackedParameter<Labels>("labels")),
  lockSources_(iPS.getUntrackedParameter<bool>("lockSources")) {
  reg.watchPreSourceConstruction(this,&LockService::preSourceConstruction);
  reg.watchPostSourceConstruction(this,&LockService::postSourceConstruction);

  reg.watchPreSource(this,&LockService::preSource);
  reg.watchPostSource(this,&LockService::postSource);

  reg.watchPreModuleEvent(this,&LockService::preModule);
  reg.watchPostModuleEvent(this,&LockService::postModule);

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
      lock_.lock();
    }
}

void LockService::postSourceConstruction(ModuleDescription const&) {
  lock_.unlock();
}

void LockService::preSource() {
  if(lockSources_) {
    lock_.lock();
  }
}

void LockService::postSource() {
  lock_.unlock();
}

void LockService::preModule(StreamContext const&, ModuleCallingContext const& iContext) {
  if(!labels_.empty() && find(labels_.begin(), labels_.end(), iContext.moduleDescription()->moduleLabel()) != labels_.end()) {
    lock_.lock();
  }
}

void LockService::postModule(StreamContext const&, ModuleCallingContext const& iContext) {
  if(!labels_.empty() && find(labels_.begin(), labels_.end(), iContext.moduleDescription()->moduleLabel()) != labels_.end()) {
    lock_.unlock();
  }
}

