// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageLogger
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  W. Brown, M. Fischler
//         Created:  Fri Nov 11 16:42:39 CST 2005
//
// Change log
//
// 1  mf  5/12/06	In ctor, MessageDrop::debugEnabled is set to a
//			sensible value in case action happens before modules
//			are entered.  If any modules enable debugs, such
//			LogDebug messages are not immediately discarded
//			(though they might be filtered at the server side).
//
// 2  mf  5/27/06	In preEventProcessing, change the syntax for
//			runEvent from 1/23 to Run: 1 Event: 23
//
// 3 mf   6/27/06	PreModuleConstruction and PreSourceConstruction get
//			correct module name
//
// 4 mf   6/27/06	Between events the run/event is previous one
//
// 5  mf  3/30/07	Support for --jobreport option
//
// 6 mf   6/6/07	Remove the catches for forgiveness of tracked
//			parameters 
//
// 7 mf   6/19/07	Support for --jobreport option
//
// 8 wmtan 6/25/07	Enable suppression for sources, just as for modules
//
// 9 mf   7/25/07	Modify names of the MessageLoggerQ methods, eg MLqLOG
//
//10 mf   6/18/07	Insert into the PostEndJob a possible SummarizeInJobReport
//
//11 mf   3/18/09	Fix wrong-sense test establishing anyDebugEnabled_
//
//12 mf   5/19/09	MessageService PSet Validation
//
//13 mf   5/26/09	Get parameters without throwing since validation 
//			will point out any problems and throw at that point
//
//14 mf   7/1/09	Establish module name and set up enables/suppresses
//			for all possible calls supplying module descriptor
//
//14 mf   7/1/09	Establish pseudo-module name and set up 
//			enables/suppresses for other calls from framework
//15 mf   9/8/09	Clean up erroneous assignments of some callbacks
//			for specific watch routines (eg PreXYZ called postXYZ)
//
//16 mf   9/8/09	Eliminate caching by descriptor address during ctor
//			phases (since addresses are not yet permanent then)
//
//17 mf   11/2/10	Move preparation of module out to MessageDrop methods
//   crj		which will only be called if a message is actually 
//			issued.  Caching of the work is done within MessageDrop
//			so that case of many messages in a module is still fast.
//
//18 mf	  11/2/10	Eliminated curr_module, since it was only being used
//			as a local variable for preparation of name (never
//			used to transfer info between functions) and change
//			17 obviates its need.
//
// 19 mf 11/30/10	Add a messageDrop->snapshot() when establishing
//    crj		module ctors, to cure bug 75836.
//
// 20 fwyzard 7/06/11   Add support fro dropping LogError messages
//                      on a per-module basis (needed at HLT)

// system include files
// user include files

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageService/interface/MessageLogger.h"
#include "FWCore/MessageService/interface/MessageServicePSetValidation.h"

#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"


#include <sstream>

using namespace edm;
using namespace edm::service;

namespace {
  char const* const s_globalTransitionNames[] = {
    "@beginJob",
    "@beginRun",
    "@beginLumi",
    "@endLumi",
    "@endRun",
    "@endJob",
    "@writeRun",
    "@writeLumi"
  };
  
  char const* const s_streamTransitionNames[] = {
    "@beginStream",
    "@streamBeginRun",
    "@streamBeginLumi",
    "",//event
    "@streamEndLumi",
    "@streamEndRun",
    "@endStream",
  };
}

namespace edm {
namespace service {

bool edm::service::MessageLogger::anyDebugEnabled_                    = false;
bool edm::service::MessageLogger::everyDebugEnabled_                  = false;
bool edm::service::MessageLogger::fjrSummaryRequested_                = false;
  
//
// constructors and destructor
//
edm::service::MessageLogger::
MessageLogger( ParameterSet const & iPS
             , ActivityRegistry   & iRegistry
                            )
        : debugEnabled_(false)
	, messageServicePSetHasBeenValidated_(false)
	, messageServicePSetValidatationResults_() 
	, nonModule_debugEnabled(false)
	, nonModule_infoEnabled(true)
	, nonModule_warningEnabled(true)
	, nonModule_errorEnabled(true)                          // change log 20
{
  // prepare cfg validation string for later use
  MessageServicePSetValidation validator;
  messageServicePSetValidatationResults_ = validator(iPS);	// change log 12
  
  typedef std::vector<std::string> vString;
  vString empty_vString;
  vString debugModules;
  vString suppressDebug;
  vString suppressInfo;
  vString suppressWarning;
  vString suppressError;                                        // change log 20

  try {								// change log 13
    // decide whether a summary should be placed in job report
    fjrSummaryRequested_ = 
    	  iPS.getUntrackedParameter<bool>("messageSummaryToJobReport", false);

    // grab list of debug-enabled modules
    debugModules = 
    	  iPS.getUntrackedParameter<vString>("debugModules", empty_vString);

    // grab lists of suppressLEVEL modules
    suppressDebug = 
    	  iPS.getUntrackedParameter<vString>("suppressDebug", empty_vString);

    suppressInfo = 
    	  iPS.getUntrackedParameter<vString>("suppressInfo", empty_vString);

    suppressWarning = 
    	  iPS.getUntrackedParameter<vString>("suppressWarning", empty_vString);

    suppressError =                                             // change log 20
    	  iPS.getUntrackedParameter<vString>("suppressError", empty_vString);
  } catch (cms::Exception& e) {					// change log 13
  }
  
  // Use these lists to prepare a map to use in tracking suppression 

  // Do suppressDebug first and suppressError last to get proper order
  for( vString::const_iterator it  = suppressDebug.begin();
                               it != suppressDebug.end(); ++it ) {
    suppression_levels_[*it] = ELseverityLevel::ELsev_success;
  }
  
  for( vString::const_iterator it  = suppressInfo.begin();
                               it != suppressInfo.end(); ++it ) {
    suppression_levels_[*it] = ELseverityLevel::ELsev_info;
  }
  
  for( vString::const_iterator it  = suppressWarning.begin();
                               it != suppressWarning.end(); ++it ) {
    suppression_levels_[*it] = ELseverityLevel::ELsev_warning;
  }

  for( vString::const_iterator it  = suppressError.begin();     // change log 20
                               it != suppressError.end(); ++it ) {
    suppression_levels_[*it] = ELseverityLevel::ELsev_error;
  }
  
  // set up for tracking whether current module is debug-enabled 
  // (and info-enabled and warning-enabled)
  if ( debugModules.empty()) {
    anyDebugEnabled_ = false;					// change log 11
    MessageDrop::instance()->debugEnabled = false;		// change log 1
  } else {
    anyDebugEnabled_ = true;					// change log 11
    MessageDrop::instance()->debugEnabled = false;
    // this will be over-ridden when specific modules are entered
  }

  // if ( debugModules.empty()) anyDebugEnabled_ = true; // wrong; change log 11
  for( vString::const_iterator it  = debugModules.begin();
                               it != debugModules.end(); ++it ) {
    if (*it == "*") { 
        everyDebugEnabled_ = true;
      } else {
        debugEnabledModules_.insert(*it); 
      }
  }
  
  								// change log 7
  std::string jm = edm::MessageDrop::jobMode; 
  std::string * jm_p = new std::string(jm);
  MessageLoggerQ::MLqMOD( jm_p ); 				// change log 9
  
  MessageLoggerQ::MLqCFG( new ParameterSet(iPS) );		// change log 9

  iRegistry.watchPreallocate([this](edm::service::SystemBounds const& iBounds){
    //reserve the proper amount of space to record the transition info
    this->transitionInfoCache_.resize(iBounds.maxNumberOfStreams()
                                  +iBounds.maxNumberOfConcurrentLuminosityBlocks()
                                  +iBounds.maxNumberOfConcurrentRuns());
    lumiInfoBegin_ = iBounds.maxNumberOfStreams();
    runInfoBegin_= lumiInfoBegin_+iBounds.maxNumberOfConcurrentLuminosityBlocks();
  });
  
  iRegistry.watchPostBeginJob(this,&MessageLogger::postBeginJob);
  iRegistry.watchPostEndJob(this,&MessageLogger::postEndJob);
  iRegistry.watchJobFailure(this,&MessageLogger::jobFailure);	// change log 14

  iRegistry.watchPreModuleConstruction(this,&MessageLogger::preModuleConstruction);
  iRegistry.watchPostModuleConstruction(this,&MessageLogger::postModuleConstruction);
								// change log 3

  iRegistry.watchPreSourceConstruction(this,&MessageLogger::preSourceConstruction);
  iRegistry.watchPostSourceConstruction(this,&MessageLogger::postSourceConstruction);
								// change log 3

  iRegistry.watchPreModuleEvent(this,&MessageLogger::preModuleEvent);
  iRegistry.watchPostModuleEvent(this,&MessageLogger::postModuleEvent);

  iRegistry.watchPreSource(this,&MessageLogger::preSource);
  iRegistry.watchPostSource(this,&MessageLogger::postSource);
							// change log 14:
  iRegistry.watchPreSourceRun(this,&MessageLogger::preSource);
  iRegistry.watchPostSourceRun(this,&MessageLogger::postSource);
  iRegistry.watchPreSourceLumi(this,&MessageLogger::preSource);
  iRegistry.watchPostSourceLumi(this,&MessageLogger::postSource);
  iRegistry.watchPreOpenFile(this,&MessageLogger::preFile);
  iRegistry.watchPostOpenFile(this,&MessageLogger::postFile);
  iRegistry.watchPreCloseFile(this,&MessageLogger::preFileClose);
  iRegistry.watchPostCloseFile(this,&MessageLogger::postFile);
  
							// change log 13:
							// change log 15
  iRegistry.watchPreModuleBeginJob(this,&MessageLogger::preModuleBeginJob);
  iRegistry.watchPostModuleBeginJob(this,&MessageLogger::postModuleBeginJob);
  iRegistry.watchPreModuleEndJob(this,&MessageLogger::preModuleEndJob);
  iRegistry.watchPostModuleEndJob(this,&MessageLogger::postModuleEndJob);
  
  iRegistry.watchPreModuleStreamBeginRun(this,&MessageLogger::preModuleStreamBeginRun);
  iRegistry.watchPostModuleStreamBeginRun(this,&MessageLogger::postModuleStreamBeginRun);
  iRegistry.watchPreModuleStreamEndRun(this,&MessageLogger::preModuleStreamEndRun);
  iRegistry.watchPostModuleStreamEndRun(this,&MessageLogger::postModuleStreamEndRun);
  iRegistry.watchPreModuleStreamBeginLumi(this,&MessageLogger::preModuleStreamBeginLumi);
  iRegistry.watchPostModuleStreamBeginLumi(this,&MessageLogger::postModuleStreamBeginLumi);
  iRegistry.watchPreModuleStreamEndLumi(this,&MessageLogger::preModuleStreamEndLumi);
  iRegistry.watchPostModuleStreamEndLumi(this,&MessageLogger::postModuleStreamEndLumi);

  iRegistry.watchPreModuleGlobalBeginRun(this,&MessageLogger::preModuleGlobalBeginRun);
  iRegistry.watchPostModuleGlobalBeginRun(this,&MessageLogger::postModuleGlobalBeginRun);
  iRegistry.watchPreModuleGlobalEndRun(this,&MessageLogger::preModuleGlobalEndRun);
  iRegistry.watchPostModuleGlobalEndRun(this,&MessageLogger::postModuleGlobalEndRun);
  iRegistry.watchPreModuleGlobalBeginLumi(this,&MessageLogger::preModuleGlobalBeginLumi);
  iRegistry.watchPostModuleGlobalBeginLumi(this,&MessageLogger::postModuleGlobalBeginLumi);
  iRegistry.watchPreModuleGlobalEndLumi(this,&MessageLogger::preModuleGlobalEndLumi);
  iRegistry.watchPostModuleGlobalEndLumi(this,&MessageLogger::postModuleGlobalEndLumi);

  iRegistry.watchPreEvent(this,&MessageLogger::preEvent);
  iRegistry.watchPostEvent(this,&MessageLogger::postEvent);

  iRegistry.watchPreStreamBeginRun(this,&MessageLogger::preStreamBeginRun);
  iRegistry.watchPostStreamBeginRun(this,&MessageLogger::postStreamBeginRun);
  iRegistry.watchPreStreamEndRun(this,&MessageLogger::preStreamEndRun);
  iRegistry.watchPostStreamEndRun(this,&MessageLogger::postStreamEndRun);
  iRegistry.watchPreStreamBeginLumi(this,&MessageLogger::preStreamBeginLumi);
  iRegistry.watchPostStreamBeginLumi(this,&MessageLogger::postStreamBeginLumi);
  iRegistry.watchPreStreamEndLumi(this,&MessageLogger::preStreamEndLumi);
  iRegistry.watchPostStreamEndLumi(this,&MessageLogger::postStreamEndLumi);

  iRegistry.watchPreGlobalBeginRun(this,&MessageLogger::preGlobalBeginRun);
  iRegistry.watchPostGlobalBeginRun(this,&MessageLogger::postGlobalBeginRun);
  iRegistry.watchPreGlobalEndRun(this,&MessageLogger::preGlobalEndRun);
  iRegistry.watchPostGlobalEndRun(this,&MessageLogger::postGlobalEndRun);
  iRegistry.watchPreGlobalBeginLumi(this,&MessageLogger::preGlobalBeginLumi);
  iRegistry.watchPostGlobalBeginLumi(this,&MessageLogger::postGlobalBeginLumi);
  iRegistry.watchPreGlobalEndLumi(this,&MessageLogger::preGlobalEndLumi);
  iRegistry.watchPostGlobalEndLumi(this,&MessageLogger::postGlobalEndLumi);

  iRegistry.watchPrePathEvent(this,&MessageLogger::prePathEvent);
  iRegistry.watchPostPathEvent(this,&MessageLogger::postPathEvent);

} // ctor

//
// Shared helper routines for establishing module name and enabling behavior
//

void
MessageLogger::establishModule(ModuleDescription const & desc,
                               const char * whichPhase)	// ChangeLog 13, 17
{
  MessageDrop* messageDrop = MessageDrop::instance();
  nonModule_debugEnabled   = messageDrop->debugEnabled;
  nonModule_infoEnabled    = messageDrop->infoEnabled;
  nonModule_warningEnabled = messageDrop->warningEnabled;
  nonModule_errorEnabled   = messageDrop->errorEnabled;         // change log 20

  // std::cerr << "establishModule( " << desc.moduleName() << ")\n";
  // Change Log 17
  messageDrop->setModuleWithPhase(desc.moduleName(), desc.moduleLabel(), 
                                  desc.id(), whichPhase );
  // Removed caching per change 17 - caching is now done in MessageDrop.cc
  // in theContext() method, and only happens if a message is actually issued.
  
  if (!anyDebugEnabled_) {
    messageDrop->debugEnabled = false;
  } else if (everyDebugEnabled_) {
    messageDrop->debugEnabled = true;
  } else {
    messageDrop->debugEnabled = 
    			debugEnabledModules_.count(desc.moduleLabel());
  }

  auto it = suppression_levels_.find(desc.moduleLabel());
  if ( it != suppression_levels_.end() ) {
    messageDrop->debugEnabled  = messageDrop->debugEnabled 
                                           && (it->second < ELseverityLevel::ELsev_success );
    messageDrop->infoEnabled    = (it->second < ELseverityLevel::ELsev_info );
    messageDrop->warningEnabled = (it->second < ELseverityLevel::ELsev_warning );
    messageDrop->errorEnabled   = (it->second < ELseverityLevel::ELsev_error );
  } else {
    messageDrop->infoEnabled    = true;
    messageDrop->warningEnabled = true;
    messageDrop->errorEnabled   = true;
  }
} // establishModule

  void
  MessageLogger::establishModule(unsigned int transitionIndex,
                                 ModuleCallingContext const & mod,
                                 const char * whichPhase)	// ChangeLog 13, 17
  {
    MessageDrop* messageDrop = MessageDrop::instance();
    nonModule_debugEnabled   = messageDrop->debugEnabled;
    nonModule_infoEnabled    = messageDrop->infoEnabled;
    nonModule_warningEnabled = messageDrop->warningEnabled;
    nonModule_errorEnabled   = messageDrop->errorEnabled;         // change log 20
    
    // std::cerr << "establishModule( " << desc.moduleName() << ")\n";
    // Change Log 17
    auto const desc = mod.moduleDescription();
    messageDrop->runEvent = transitionInfoCache_[transitionIndex];
    messageDrop->setModuleWithPhase(desc->moduleName(), desc->moduleLabel(),
                                    desc->id(), whichPhase );
    // Removed caching per change 17 - caching is now done in MessageDrop.cc
    // in theContext() method, and only happens if a message is actually issued.
    
    if (!anyDebugEnabled_) {
      messageDrop->debugEnabled = false;
    } else if (everyDebugEnabled_) {
      messageDrop->debugEnabled = true;
    } else {
      messageDrop->debugEnabled =
      debugEnabledModules_.count(desc->moduleLabel());
    }
    
    auto it = suppression_levels_.find(desc->moduleLabel());
    if ( it != suppression_levels_.end() ) {
      messageDrop->debugEnabled  = messageDrop->debugEnabled
      && (it->second < ELseverityLevel::ELsev_success );
      messageDrop->infoEnabled    = (it->second < ELseverityLevel::ELsev_info );
      messageDrop->warningEnabled = (it->second < ELseverityLevel::ELsev_warning );
      messageDrop->errorEnabled   = (it->second < ELseverityLevel::ELsev_error );
    } else {
      messageDrop->infoEnabled    = true;
      messageDrop->warningEnabled = true;
      messageDrop->errorEnabled   = true;
    }
  } // establishModule

  
void
MessageLogger::unEstablishModule(ModuleDescription const & /*unused*/, 
			         const char*  state)
{
      // std::cerr << "unestablishModule( " << desc.moduleName() << ") "
      //           << "state = " << *state << "\n";

  MessageDrop* messageDrop = MessageDrop::instance();
  messageDrop->setSinglet( state ); 			// Change Log 17	
  messageDrop->debugEnabled   = nonModule_debugEnabled;
  messageDrop->infoEnabled    = nonModule_infoEnabled;
  messageDrop->warningEnabled = nonModule_warningEnabled;
  messageDrop->errorEnabled   = nonModule_errorEnabled; // change log 20
}

  void
  MessageLogger::unEstablishModule(ModuleCallingContext const & mod,
                                   const char*  state)
  {
    //Need to reset to what was previously being used on this thread
    auto previous = mod.previousModuleOnThread();
    if(previous) {
      //need to know if we are in a global or stream context
      auto top = previous->getTopModuleCallingContext();
      assert(nullptr != top);
      if (ParentContext::Type::kGlobal == top->type()) {
        auto globalContext = top->globalContext();
        assert(nullptr != globalContext);
        auto tran = globalContext->transition();
        if(tran == GlobalContext::Transition::kBeginLuminosityBlock or
           tran == GlobalContext::Transition::kEndLuminosityBlock) {
          establishModule(lumiInfoBegin_+globalContext->luminosityBlockIndex(),
                          *previous,s_globalTransitionNames[static_cast<int>(tran)]);
        } else {
          establishModule(runInfoBegin_+globalContext->runIndex(),
                          *previous,s_globalTransitionNames[static_cast<int>(tran)]);
        }
      } else {
        auto stream = previous->getStreamContext();
        establishModule(stream->streamID().value(),*previous,s_streamTransitionNames[static_cast<int>(stream->transition())]);
      }
    } else {
      MessageDrop* messageDrop = MessageDrop::instance();
      messageDrop->setSinglet( state ); 			// Change Log 17
      messageDrop->debugEnabled   = nonModule_debugEnabled;
      messageDrop->infoEnabled    = nonModule_infoEnabled;
      messageDrop->warningEnabled = nonModule_warningEnabled;
      messageDrop->errorEnabled   = nonModule_errorEnabled; // change log 20
    }

    // std::cerr << "unestablishModule( " << desc.moduleName() << ") "
    //           << "state = " << *state << "\n";
    
  }

void
MessageLogger::establish(const char* state)
{
  MessageDrop* messageDrop = MessageDrop::instance();
  messageDrop->setSinglet( state ); 			// Change Log 17	
   if (!anyDebugEnabled_) {
    messageDrop->debugEnabled = false;
  } else if (everyDebugEnabled_) {
    messageDrop->debugEnabled = true;
  } else {
    messageDrop->debugEnabled = 
    		debugEnabledModules_.count(state);	// change log 8
  }
  std::map<const std::string,ELseverityLevel>::const_iterator it =
       suppression_levels_.find(state);		// change log 8
  if ( it != suppression_levels_.end() ) {
    messageDrop->debugEnabled  = messageDrop->debugEnabled
                                           && (it->second < ELseverityLevel::ELsev_success );
    messageDrop->infoEnabled    = (it->second < ELseverityLevel::ELsev_info );
    messageDrop->warningEnabled = (it->second < ELseverityLevel::ELsev_warning );
    messageDrop->errorEnabled   = (it->second < ELseverityLevel::ELsev_error );
  } else {
    messageDrop->infoEnabled    = true;
    messageDrop->warningEnabled = true;
    messageDrop->errorEnabled   = true;
  }
}

void
MessageLogger::unEstablish(const char* state)
{
  MessageDrop::instance()->setSinglet( state ); 	// Change Log 17	
}

//
// callbacks that need to establish the module, and their counterparts
//

void
MessageLogger::preModuleConstruction(const ModuleDescription& desc)
{
  if (!messageServicePSetHasBeenValidated_) {			// change log 12
    if (!messageServicePSetValidatationResults_.empty() ) {
      throw ( edm::Exception 
                   ( edm::errors::Configuration
                   , messageServicePSetValidatationResults_ 
	           )                                         );
    }
    messageServicePSetHasBeenValidated_ = true;
  }
  establishModule (desc,"@ctor");				// ChangeLog 16
}
void MessageLogger::postModuleConstruction(const ModuleDescription& iDescription)
{ //it is now guaranteed that this will be called even if the module throws
  unEstablishModule (iDescription, "AfterModConstruction"); }

void
MessageLogger::preModuleBeginJob(const ModuleDescription& desc)
{
  establishModule (desc,"@beginJob");				// ChangeLog 13
}
void MessageLogger::postModuleBeginJob(const ModuleDescription& iDescription)
{ unEstablishModule (iDescription, "AfterModBeginJob"); }

void
MessageLogger::preSourceConstruction(const ModuleDescription& desc)
{
  if (!messageServicePSetHasBeenValidated_) {			// change log 12
    if (!messageServicePSetValidatationResults_.empty() ) {
      throw ( edm::Exception 
                   ( edm::errors::Configuration
                   , messageServicePSetValidatationResults_ 
	           )                                         );
    }
    messageServicePSetHasBeenValidated_ = true;
  } 
  establishModule(desc,"@sourceConstruction");		// ChangeLog 16
}
void MessageLogger::postSourceConstruction(const ModuleDescription& iDescription)
{ unEstablishModule (iDescription, "AfterSourceConstruction"); }

void
MessageLogger::preModuleStreamBeginRun(StreamContext const& stream, ModuleCallingContext const& mod)
{
  establishModule (stream.streamID().value(),mod,
                   s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kBeginRun)]);
}
void MessageLogger::postModuleStreamBeginRun(StreamContext const& stream, ModuleCallingContext const& mod)
{ unEstablishModule (mod, "AfterModStreamBeginRun"); }

void
MessageLogger::preModuleStreamBeginLumi(StreamContext const& stream, ModuleCallingContext const& mod)
{
  establishModule (stream.streamID().value(),mod,
                   s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kBeginLuminosityBlock)]);
}
void MessageLogger::postModuleStreamBeginLumi(StreamContext const& stream, ModuleCallingContext const& mod)
{ unEstablishModule (mod, "AfterModStreamBeginLumi"); }

void
MessageLogger::preModuleEvent(StreamContext const& stream, ModuleCallingContext const& mod)
{
  establishModule (stream.streamID().value(),mod,
                   s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kEvent)]);
}

void MessageLogger::postModuleEvent(StreamContext const& stream, ModuleCallingContext const& mod)
{
  unEstablishModule(mod,"PostModuleEvent");
}

void
MessageLogger::preModuleStreamEndLumi(StreamContext const& stream, ModuleCallingContext const& mod)
{
  establishModule (stream.streamID().value(),mod,
                   s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kEndLuminosityBlock)]);
}
void MessageLogger::postModuleStreamEndLumi(StreamContext const& stream, ModuleCallingContext const& mod)
{ unEstablishModule (mod, "AfterModStreamEndLumi"); }

void
MessageLogger::preModuleStreamEndRun(StreamContext const& stream, ModuleCallingContext const& mod)
{
  establishModule (stream.streamID().value(),mod,
                   s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kEndRun)]);				// ChangeLog 13
}
void MessageLogger::postModuleStreamEndRun(StreamContext const& stream, ModuleCallingContext const& mod)
{ unEstablishModule (mod, "AfterModStreamEndRun"); }

  //Global
void
MessageLogger::preModuleGlobalBeginRun(GlobalContext const& context, ModuleCallingContext const& mod)
{
  establishModule (runInfoBegin_+ context.runIndex().value(),mod,
                   s_globalTransitionNames[static_cast<int>(GlobalContext::Transition::kBeginRun)]);
}
void MessageLogger::postModuleGlobalBeginRun(GlobalContext const& context, ModuleCallingContext const& mod)
{ unEstablishModule (mod, "AfterModGlobalBeginRun"); }

void
MessageLogger::preModuleGlobalBeginLumi(GlobalContext const& context, ModuleCallingContext const& mod)
{
  establishModule (lumiInfoBegin_+ context.luminosityBlockIndex().value(),mod,
                   s_globalTransitionNames[static_cast<int>(GlobalContext::Transition::kBeginLuminosityBlock)]);
}
void MessageLogger::postModuleGlobalBeginLumi(GlobalContext const& stream, ModuleCallingContext const& mod)
{ unEstablishModule (mod, "AfterModGlobalBeginLumi"); }

void
MessageLogger::preModuleGlobalEndLumi(GlobalContext const& context, ModuleCallingContext const& mod)
{
  establishModule (lumiInfoBegin_+ context.luminosityBlockIndex().value(),mod,
                   s_globalTransitionNames[static_cast<int>(GlobalContext::Transition::kEndLuminosityBlock)]);
}
void MessageLogger::postModuleGlobalEndLumi(GlobalContext const& stream, ModuleCallingContext const& mod)
{ unEstablishModule (mod, "AfterModGlobalEndLumi"); }

void
MessageLogger::preModuleGlobalEndRun(GlobalContext const& context, ModuleCallingContext const& mod)
{
  establishModule (runInfoBegin_+ context.runIndex().value(), mod,
                   s_globalTransitionNames[static_cast<int>(GlobalContext::Transition::kEndRun)]);				// ChangeLog 13
}
void MessageLogger::postModuleGlobalEndRun(GlobalContext const& stream, ModuleCallingContext const& mod)
{ unEstablishModule (mod, "AfterModGlobalEndRun"); }
  
void
MessageLogger::preModuleEndJob(const ModuleDescription& desc)
{
  establishModule (desc,"@endJob");				// ChangeLog 13
}
void MessageLogger::postModuleEndJob(const ModuleDescription& iDescription)
{ unEstablishModule (iDescription, "AfterModEndJob"); }

//
// callbacks that don't know about the module
//

void
MessageLogger::postBeginJob()
{
  MessageDrop::instance()->runEvent = "BeforeEvents";  
  MessageDrop::instance()->setSinglet("AfterBeginJob");     // Change Log 17	
}

void
MessageLogger::preSource()
{
  establish("source");
}
void MessageLogger::postSource()
{ unEstablish("AfterSource"); }

void MessageLogger::preFile( std::string const &, bool )
{  establish("file_open"); }
void MessageLogger::preFileClose( std::string const &, bool )
{  establish("file_close"); }
void MessageLogger::postFile( std::string const &, bool )
{ unEstablish("AfterFile"); }


void
MessageLogger::preEvent( StreamContext const& iContext)
{
  std::ostringstream ost;
  auto const& id = iContext.eventID();
  ost << "Run: " << id.run()
      << " Event: " << id.event();    			// change log 2
  assert(iContext.streamID().value()<transitionInfoCache_.size());
  transitionInfoCache_[iContext.streamID().value()]=ost.str();
  edm::MessageDrop::instance()->runEvent = ost.str();  
  edm::MessageDrop::instance()->setSinglet("PreEventProcessing");// changelog 17
  // Note - module name had not been set here  Similarly in other places where
  // RunEvent carries the new information; we add setSinglet for module name.
}

void
MessageLogger::postEvent(StreamContext const&,const Event&, const EventSetup&)
{
  edm::MessageDrop::instance()->runEvent = "PostProcessEvent";  
}

  void
  MessageLogger::preStreamBeginRun( StreamContext const& iContext)	// change log 14
  {
    std::ostringstream ost;
    ost << "Run: " << iContext.eventID().run();
    transitionInfoCache_[iContext.streamID().value()]=ost.str();
    edm::MessageDrop::instance()->runEvent = ost.str();
    edm::MessageDrop::instance()->setSinglet("PreBeginRun");	// changelog 17
  }
  void MessageLogger::postStreamBeginRun(StreamContext const&)
  {
    edm::MessageDrop::instance()->runEvent = "PostBeginRun";
    edm::MessageDrop::instance()->setSinglet("PostBeginRun");	// changelog 17
                                                              // Note - module name had not been set here
  }

  void
  MessageLogger::preStreamEndRun( StreamContext const& iContext)
  {
    std::ostringstream ost;
    ost << "End Run: " << iContext.eventID().run();
    transitionInfoCache_[iContext.streamID().value()]=ost.str();
    edm::MessageDrop::instance()->runEvent = ost.str();
    edm::MessageDrop::instance()->setSinglet("PreEndRun");	// changelog 17
  }
  
  void MessageLogger::postStreamEndRun(StreamContext const&, const Run&, const EventSetup&)
  {
    edm::MessageDrop::instance()->runEvent = "PostEndRun";
    edm::MessageDrop::instance()->setSinglet("PostEndRun");	// changelog 17
  }
  
  void
  MessageLogger::preStreamBeginLumi( StreamContext const& iContext)
  {
    std::ostringstream ost;
    auto const& id = iContext.eventID();
    ost << "Run: " << id.run() << " Lumi: " << id.luminosityBlock();
    transitionInfoCache_[iContext.streamID().value()]=ost.str();
    edm::MessageDrop::instance()->runEvent = ost.str();
    edm::MessageDrop::instance()->setSinglet("PreBeginLumi");	// changelog 17
  }
  
  void MessageLogger::postStreamBeginLumi(StreamContext const&)
  {
    edm::MessageDrop::instance()->runEvent = "PostBeginLumi";
    edm::MessageDrop::instance()->setSinglet("PostBeginLumi");	// changelog 17
  }
  
  void
  MessageLogger::preStreamEndLumi( StreamContext const& iContext)
  {
    std::ostringstream ost;
    auto const& id = iContext.eventID();
    ost << "Run: " << id.run() << " Lumi: " << id.luminosityBlock();
    transitionInfoCache_[iContext.streamID().value()]=ost.str();
    edm::MessageDrop::instance()->runEvent = ost.str();
    edm::MessageDrop::instance()->setSinglet("PreEndLumi");	// changelog 17
  }
  void MessageLogger::postStreamEndLumi(StreamContext const&, const LuminosityBlock&, const EventSetup&)
  {
    edm::MessageDrop::instance()->runEvent = "PostEndLumi";
    edm::MessageDrop::instance()->setSinglet("PostEndLumi");	// changelog 17
  }

  
void
MessageLogger::preGlobalBeginRun( GlobalContext const& iContext)	// change log 14
{
  std::ostringstream ost;
  ost << "Run: " << iContext.luminosityBlockID().run();
  transitionInfoCache_[runInfoBegin_+iContext.runIndex()]=ost.str();
  edm::MessageDrop::instance()->runEvent = ost.str();  
  edm::MessageDrop::instance()->setSinglet("PreBeginRun");	// changelog 17
}
void MessageLogger::postGlobalBeginRun(GlobalContext const&)
{ 
  edm::MessageDrop::instance()->runEvent = "PostBeginRun"; 
  edm::MessageDrop::instance()->setSinglet("PostBeginRun");	// changelog 17
  // Note - module name had not been set here
}

void
MessageLogger::prePathEvent( StreamContext const&, PathContext const& iPath)	// change log 14
{
  edm::MessageDrop::instance()->setPath( "PreProcPath ", iPath.pathName());
  								// change log 17
}

void MessageLogger::postPathEvent(StreamContext const&, PathContext const&, HLTPathStatus const&)
{ 
  edm::MessageDrop::instance()->setSinglet("PostProcessPath");	// changelog 17
}

void
MessageLogger::preGlobalEndRun( GlobalContext const& iContext)
{
  std::ostringstream ost;
  ost << "End Run: " << iContext.luminosityBlockID().run();
  transitionInfoCache_[runInfoBegin_+iContext.runIndex()]=ost.str();
  edm::MessageDrop::instance()->runEvent = ost.str();
  edm::MessageDrop::instance()->setSinglet("PreEndRun");	// changelog 17
}

void MessageLogger::postGlobalEndRun(GlobalContext const&, const Run&, const EventSetup&)
{ 
  edm::MessageDrop::instance()->runEvent = "PostEndRun"; 
  edm::MessageDrop::instance()->setSinglet("PostEndRun");	// changelog 17
}

void
MessageLogger::preGlobalBeginLumi( GlobalContext const& iContext)
{
  std::ostringstream ost;
  auto const& id = iContext.luminosityBlockID();
  ost << "Run: " << id.run() << " Lumi: " << id.luminosityBlock();
  transitionInfoCache_[lumiInfoBegin_+iContext.luminosityBlockIndex()]=ost.str();
  edm::MessageDrop::instance()->runEvent = ost.str();
  edm::MessageDrop::instance()->setSinglet("PreBeginLumi");	// changelog 17
}

void MessageLogger::postGlobalBeginLumi(GlobalContext const&)
{ 
  edm::MessageDrop::instance()->runEvent = "PostBeginLumi"; 
  edm::MessageDrop::instance()->setSinglet("PostBeginLumi");	// changelog 17
}

void
MessageLogger::preGlobalEndLumi( GlobalContext const& iContext)
{
  std::ostringstream ost;
  auto const& id = iContext.luminosityBlockID();
  ost << "Run: " << id.run() << " Lumi: " << id.luminosityBlock();
  transitionInfoCache_[lumiInfoBegin_+iContext.luminosityBlockIndex()]=ost.str();
  edm::MessageDrop::instance()->runEvent = ost.str();
  edm::MessageDrop::instance()->setSinglet("PreEndLumi");	// changelog 17
}
void MessageLogger::postGlobalEndLumi(GlobalContext const&, const LuminosityBlock&, const EventSetup&)
{ 
  edm::MessageDrop::instance()->runEvent = "PostEndLumi"; 
  edm::MessageDrop::instance()->setSinglet("PostEndLumi");	// changelog 17
}

void
MessageLogger::postEndJob()
{
  SummarizeInJobReport();     // Put summary info into Job Rep  // change log 10
  MessageLoggerQ::MLqSUM ( ); // trigger summary info.		// change log 9
}

void
MessageLogger::jobFailure()
{
  MessageDrop* messageDrop = MessageDrop::instance();
  messageDrop->setSinglet("jobFailure");
  SummarizeInJobReport();     // Put summary info into Job Rep  // change log 10
  MessageLoggerQ::MLqSUM ( ); // trigger summary info.		// change log 9
}


//
// Other methods
//

void
MessageLogger::SummarizeInJobReport() {
  if ( fjrSummaryRequested_ ) { 
    std::map<std::string, double> * smp = new std::map<std::string, double> ();
    MessageLoggerQ::MLqJRS ( smp );
    Service<JobReport> reportSvc;
    reportSvc->reportMessageInfo(*smp);
    delete smp;
  } 
}

} // end of namespace service  
} // end of namespace edm  
