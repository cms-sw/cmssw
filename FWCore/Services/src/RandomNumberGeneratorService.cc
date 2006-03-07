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
// $Id$
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

   iRegistry.watchPreProcessEvent(this,&RandomNumberGeneratorService::preEventProcessing);
   iRegistry.watchPostProcessEvent(this,&RandomNumberGeneratorService::postEventProcessing);
   
   iRegistry.watchPreModule(this,&RandomNumberGeneratorService::preModule);
   iRegistry.watchPostModule(this,&RandomNumberGeneratorService::postModule);
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
   presentGen_ = labelToSeed_.find(iDesc.moduleLabel_);
   unknownLabel_=iDesc.moduleLabel_;
}
void 
RandomNumberGeneratorService::postModuleConstruction(const ModuleDescription&)
{
   presentGen_ = labelToSeed_.end();
   unknownLabel_=std::string();
}

void 
RandomNumberGeneratorService::postBeginJob()
{
   //finished begin run so waiting for first event and the source will be the first one called
   presentGen_ = labelToSeed_.find(sourceLabel);
   unknownLabel_=sourceLabel;
}
void 
RandomNumberGeneratorService::postEndJob()
{
   presentGen_ = labelToSeed_.end();
   unknownLabel_=std::string();
}

void 
RandomNumberGeneratorService::preEventProcessing(const edm::EventID&, const edm::Timestamp&)
{
   //finished with source and now waiting for a module
   presentGen_ = labelToSeed_.end();
   unknownLabel_=std::string();
}
void 
RandomNumberGeneratorService::postEventProcessing(const Event&, const EventSetup&)
{
   //finished processing the event so should start another one soon.  The first thing to be called will be the source
   presentGen_ = labelToSeed_.find(sourceLabel);
   unknownLabel_=sourceLabel;
}

void 
RandomNumberGeneratorService::preModule(const ModuleDescription& iDesc)
{
   presentGen_ = labelToSeed_.find(iDesc.moduleLabel_);
   unknownLabel_=iDesc.moduleLabel_;
}
void 
RandomNumberGeneratorService::postModule(const ModuleDescription&)
{
   presentGen_ = labelToSeed_.end();
   unknownLabel_=std::string();
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
