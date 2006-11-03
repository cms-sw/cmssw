

#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/GlobalMutex.h"

#include <iostream>
#include <string>

using namespace evf;
using namespace std;


MicroStateService::MicroStateService(const edm::ParameterSet& iPS, 
			 edm::ActivityRegistry& reg)
{
  
  reg.watchPostBeginJob(this,&MicroStateService::postBeginJob);
  reg.watchPostEndJob(this,&MicroStateService::postEndJob);
  
  reg.watchPreProcessEvent(this,&MicroStateService::preEventProcessing);
  reg.watchPostProcessEvent(this,&MicroStateService::postEventProcessing);
  reg.watchPreSource(this,&MicroStateService::preSource);
  reg.watchPostSource(this,&MicroStateService::postSource);
  
  reg.watchPreModule(this,&MicroStateService::preModule);
  reg.watchPostModule(this,&MicroStateService::postModule);
  microstate1_ = "BJ";
  microstate2_ = "INIT";

}


MicroStateService::~MicroStateService()
{
}

void MicroStateService::postBeginJob()
{
  microstate1_ = "BJD";
}

void MicroStateService::postEndJob()
{
  microstate1_ = "EJ";
  microstate2_ = "done";
}

void MicroStateService::preEventProcessing(const edm::EventID& iID,
				     const edm::Timestamp& iTime)
{
  microstate1_ = "PRO";
}

void MicroStateService::postEventProcessing(const edm::Event& e, const edm::EventSetup&)
{
}
void MicroStateService::preSource()
{
  microstate2_ = "IN";
}

void MicroStateService::postSource()
{
  microstate2_ = "IND";
}

void MicroStateService::preModule(const edm::ModuleDescription& desc)
{
  microstate2_ = desc.moduleLabel_;
}

void MicroStateService::postModule(const edm::ModuleDescription& desc)
{
}


