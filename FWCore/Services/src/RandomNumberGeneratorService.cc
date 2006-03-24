// -*- C++ -*-
//
// Package:     Services
// Class  :     RandomNumberGeneratorService
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Mar  7 09:43:46 EST 2006
// $Id: RandomNumberGeneratorService.cc,v 1.1 2006/03/07 19:46:37 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Services/src/RandomNumberGeneratorService.h"

#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// constants, enums and typedefs
//
using namespace edm::service;

//
// static data member definitions
//
static const std::string sourceLabel("@source");
//
// constructors and destructor
//
RandomNumberGeneratorService::RandomNumberGeneratorService(const edm::ParameterSet& iPSet,
                                                           edm::ActivityRegistry& iRegistry):
presentGen_(labelToSeed_.end())
{
   try {
      uint32_t seed = iPSet.getUntrackedParameter<uint32_t>("sourceSeed");
      labelToSeed_[sourceLabel] = seed;
   }catch(const edm::Exception&) {
      //I need to know if the parameter is there or not but it is OK for the source to be missing
   }
   
   try {
      const edm::ParameterSet& moduleSeeds = iPSet.getParameter<edm::ParameterSet>("moduleSeeds");
      //all parameters in the 'moduleSeeds' PSet are required to be uint32s and the parameter names are to match
      // module labels used in the configuration
      std::vector<std::string> names=   moduleSeeds.getParameterNames();
      for(std::vector<std::string>::iterator itName = names.begin();itName != names.end(); ++itName) {
	 uint32_t seed = moduleSeeds.getUntrackedParameter<uint32_t>(*itName);
	 labelToSeed_[*itName]=seed;
      }
   }catch(const edm::Exception&) {
      //It is OK if this is missing.
   }
   
   iRegistry.watchPostBeginJob(this,&RandomNumberGeneratorService::postBeginJob);
   iRegistry.watchPostEndJob(this,&RandomNumberGeneratorService::postEndJob);

   iRegistry.watchPreModuleConstruction(this,&RandomNumberGeneratorService::preModuleConstruction);
   iRegistry.watchPostModuleConstruction(this,&RandomNumberGeneratorService::postModuleConstruction);

   iRegistry.watchPreSourceConstruction(this,&RandomNumberGeneratorService::preSourceConstruction);
   iRegistry.watchPostSourceConstruction(this,&RandomNumberGeneratorService::postSourceConstruction);

   iRegistry.watchPreProcessEvent(this,&RandomNumberGeneratorService::preEventProcessing);
   iRegistry.watchPostProcessEvent(this,&RandomNumberGeneratorService::postEventProcessing);
   
   iRegistry.watchPreModule(this,&RandomNumberGeneratorService::preModule);
   iRegistry.watchPostModule(this,&RandomNumberGeneratorService::postModule);

   //the default for the stack is to point to the 'end' of our labels which is used to define not set
   labelStack_.push_back(labelToSeed_.end());
   unknownLabelStack_.push_back(std::string());
}

// RandomNumberGeneratorService::RandomNumberGeneratorService(const RandomNumberGeneratorService& rhs)
// {
//    // do actual copying here;
// }

//RandomNumberGeneratorService::~RandomNumberGeneratorService()
//{
//}

//
// assignment operators
//
// const RandomNumberGeneratorService& RandomNumberGeneratorService::operator=(const RandomNumberGeneratorService& rhs)
// {
//   //An exception safe implementation is
//   RandomNumberGeneratorService temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
RandomNumberGeneratorService::preModuleConstruction(const ModuleDescription& iDesc)
{
  push(iDesc.moduleLabel_);
}
void 
RandomNumberGeneratorService::postModuleConstruction(const ModuleDescription&)
{
  pop();
}

void 
RandomNumberGeneratorService::preSourceConstruction(const ModuleDescription& iDesc)
{
  push(sourceLabel);
}
void 
RandomNumberGeneratorService::postSourceConstruction(const ModuleDescription&)
{
  pop();
}

void 
RandomNumberGeneratorService::postBeginJob()
{
  //finished begin run so waiting for first event and the source will be the first one called
  push(sourceLabel);
}
void 
RandomNumberGeneratorService::postEndJob()
{
  if(unknownLabelStack_.size()!=1) {
    pop();
  }
}

void 
RandomNumberGeneratorService::preEventProcessing(const edm::EventID&, const edm::Timestamp&)
{
  //finished with source and now waiting for a module
  pop();
}
void 
RandomNumberGeneratorService::postEventProcessing(const Event&, const EventSetup&)
{
   //finished processing the event so should start another one soon.  The first thing to be called will be the source
  push(sourceLabel);
}

void 
RandomNumberGeneratorService::preModule(const ModuleDescription& iDesc)
{
  push(iDesc.moduleLabel_);
}
void 
RandomNumberGeneratorService::postModule(const ModuleDescription&)
{
  pop();
}

void
RandomNumberGeneratorService::push(const std::string& iLabel)
{
  presentGen_ = labelToSeed_.find(iLabel);
  labelStack_.push_back(presentGen_);
  
  unknownLabelStack_.push_back(iLabel);
  unknownLabel_=iLabel;
}
void
RandomNumberGeneratorService::pop()
{
  labelStack_.pop_back();
  //NOTE: algorithm is such that we always have at least one item in the stacks
  presentGen_ = labelStack_.back();
  unknownLabelStack_.pop_back();
  unknownLabel_=unknownLabelStack_.back();
}
//
// const member functions
//
uint32_t
RandomNumberGeneratorService::mySeed() const {
   if(presentGen_ == labelToSeed_.end()) {
      if(unknownLabel_ != std::string() ) {
         if( unknownLabel_ != sourceLabel) {
            throw cms::Exception("RandomNumberFailure")<<"requested a random number seed for a module with label \""<<unknownLabel_<<"\" that was not configured to use a random number."
            " Please change configuration file so that the RandomNumberGeneratorService has a random number seed for \""<<unknownLabel_<<"\"";
         }else {
            throw cms::Exception("RandomNumberFailure")<<"requested a random number seed for the source but the source was not configured to use a random number."
            " Please change configuration file so that the RandomNumberGeneratorService has a random number seed the source";}
      } else {
         throw cms::Exception("RandomNumberFailure")<<"requested a random number when no module was active.  This is not supposed to be possible."
         "  Please rerun the job using a debugger to get a traceback to show what routines were called and then send information to edm developers.";
      }
   }  
   return presentGen_->second;
}


//
// static member functions
//
