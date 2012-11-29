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

#include <sstream>

using namespace edm;
using namespace edm::service;

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
    MessageDrop::debugEnabled = false;		// change log 1
  } else {
    anyDebugEnabled_ = true;					// change log 11
    MessageDrop::debugEnabled = false;
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

  								// change log 5
  std::string jr_name = edm::MessageDrop::instance()->jobreport_name; 
  if (!jr_name.empty()) {			
    std::string * jr_name_p = new std::string(jr_name);
    MessageLoggerQ::MLqJOB( jr_name_p ); 			// change log 9
  }
  
  								// change log 7
  std::string jm = edm::MessageDrop::instance()->jobMode; 
  std::string * jm_p = new std::string(jm);
  MessageLoggerQ::MLqMOD( jm_p ); 				// change log 9
  
  MessageLoggerQ::MLqCFG( new ParameterSet(iPS) );		// change log 9

  iRegistry.watchPostBeginJob(this,&MessageLogger::postBeginJob);
  iRegistry.watchPostEndJob(this,&MessageLogger::postEndJob);
  iRegistry.watchJobFailure(this,&MessageLogger::jobFailure);	// change log 14

  iRegistry.watchPreModuleConstruction(this,&MessageLogger::preModuleConstruction);
  iRegistry.watchPostModuleConstruction(this,&MessageLogger::postModuleConstruction);
								// change log 3

  iRegistry.watchPreSourceConstruction(this,&MessageLogger::preSourceConstruction);
  iRegistry.watchPostSourceConstruction(this,&MessageLogger::postSourceConstruction);
								// change log 3

  iRegistry.watchPreModule(this,&MessageLogger::preModule);
  iRegistry.watchPostModule(this,&MessageLogger::postModule);

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
  iRegistry.watchPreModuleBeginRun(this,&MessageLogger::preModuleBeginRun);
  iRegistry.watchPostModuleBeginRun(this,&MessageLogger::postModuleBeginRun);
  iRegistry.watchPreModuleEndRun(this,&MessageLogger::preModuleEndRun);
  iRegistry.watchPostModuleEndRun(this,&MessageLogger::postModuleEndRun);
  iRegistry.watchPreModuleBeginLumi(this,&MessageLogger::preModuleBeginLumi);
  iRegistry.watchPostModuleBeginLumi(this,&MessageLogger::postModuleBeginLumi);
  iRegistry.watchPreModuleEndLumi(this,&MessageLogger::preModuleEndLumi);
  iRegistry.watchPostModuleEndLumi(this,&MessageLogger::postModuleEndLumi);

  iRegistry.watchPreProcessEvent(this,&MessageLogger::preEventProcessing);
  iRegistry.watchPostProcessEvent(this,&MessageLogger::postEventProcessing);
							// change log 14:
  iRegistry.watchPreBeginRun(this,&MessageLogger::preBeginRun);
  iRegistry.watchPostBeginRun(this,&MessageLogger::postBeginRun);
  iRegistry.watchPreEndRun(this,&MessageLogger::preEndRun); // change log 15
  iRegistry.watchPostEndRun(this,&MessageLogger::postEndRun);
  iRegistry.watchPreBeginLumi(this,&MessageLogger::preBeginLumi);
  iRegistry.watchPostBeginLumi(this,&MessageLogger::postBeginLumi);
  iRegistry.watchPreEndLumi(this,&MessageLogger::preEndLumi);
  iRegistry.watchPostEndLumi(this,&MessageLogger::postEndLumi);

  iRegistry.watchPrePathBeginRun(this,&MessageLogger::prePathBeginRun);
  iRegistry.watchPostPathBeginRun(this,&MessageLogger::postPathBeginRun);
  iRegistry.watchPrePathEndRun(this,&MessageLogger::prePathEndRun);
  iRegistry.watchPostPathEndRun(this,&MessageLogger::postPathEndRun);
  iRegistry.watchPrePathBeginLumi(this,&MessageLogger::prePathBeginLumi);
  iRegistry.watchPostPathBeginLumi(this,&MessageLogger::postPathBeginLumi);
  iRegistry.watchPrePathEndLumi(this,&MessageLogger::prePathEndLumi);
  iRegistry.watchPostPathEndLumi(this,&MessageLogger::postPathEndLumi);
  iRegistry.watchPreProcessPath(this,&MessageLogger::preProcessPath);
  iRegistry.watchPostProcessPath(this,&MessageLogger::postProcessPath);

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
  messageDrop->setModuleWithPhase( desc.moduleName(), desc.moduleLabel(), 
  				&desc, whichPhase );
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

  std::map<const std::string,ELseverityLevel>::const_iterator it =
       suppression_levels_.find(desc.moduleLabel());
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
MessageLogger::establishModuleCtor(ModuleDescription const & desc, 
			       const char* whichPhase)	// ChangeLog 16
{
  MessageDrop* messageDrop = MessageDrop::instance();
  nonModule_debugEnabled   = messageDrop->debugEnabled;
  nonModule_infoEnabled    = messageDrop->infoEnabled;
  nonModule_warningEnabled = messageDrop->warningEnabled;
  nonModule_errorEnabled   = messageDrop->errorEnabled;         // change log 20

  // std::cerr << "establishModuleCtor( " << desc.moduleName() << ")\n";
  // Change Log 17
  messageDrop->setModuleWithPhase( desc.moduleName(), desc.moduleLabel(), 
  				0, whichPhase );
  // Cannot cache the value to improve performance because addresses are 
  // not yet permanent - see change log 16.  So did not provide desc ptr.

  if (!anyDebugEnabled_) {
    messageDrop->debugEnabled = false;
  } else if (everyDebugEnabled_) {
    messageDrop->debugEnabled = true;
  } else {
    messageDrop->debugEnabled = 
    			debugEnabledModules_.count(desc.moduleLabel());
  }

  std::map<const std::string,ELseverityLevel>::const_iterator it =
       suppression_levels_.find(desc.moduleLabel());
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
  messageDrop->snapshot();				// Change Log 18 
} // establishModuleCtor

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
  establishModuleCtor (desc,"@ctor");				// ChangeLog 16
}
void MessageLogger::postModuleConstruction(const ModuleDescription& iDescription)
{ unEstablishModule (iDescription, "AfterModConstruction"); }

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
  establishModuleCtor (desc,"@sourceConstruction");		// ChangeLog 16
}
void MessageLogger::postSourceConstruction(const ModuleDescription& iDescription)
{ unEstablishModule (iDescription, "AfterSourceConstruction"); }

void
MessageLogger::preModuleBeginRun(const ModuleDescription& desc)
{
  establishModule (desc,"@beginRun");				// ChangeLog 13
}
void MessageLogger::postModuleBeginRun(const ModuleDescription& iDescription)
{ unEstablishModule (iDescription, "AfterModBeginRun"); }

void
MessageLogger::preModuleBeginLumi(const ModuleDescription& desc)
{
  establishModule (desc,"@beginLumi");				// ChangeLog 13
}
void MessageLogger::postModuleBeginLumi(const ModuleDescription& iDescription)
{ unEstablishModule (iDescription, "AfterModBeginLumi"); }

void
MessageLogger::preModule(const ModuleDescription& desc)
{
  establishModule (desc,"");					// ChangeLog 13
}
void MessageLogger::postModule(const ModuleDescription& iDescription)
{ unEstablishModule (iDescription, "PostModule"); }

void
MessageLogger::preModuleEndLumi(const ModuleDescription& desc)
{
  establishModule (desc,"@endLumi");				// ChangeLog 13
}
void MessageLogger::postModuleEndLumi(const ModuleDescription& iDescription)
{ unEstablishModule (iDescription, "AfterModEndLumi"); }

void
MessageLogger::preModuleEndRun(const ModuleDescription& desc)
{
  establishModule (desc,"@endRun");				// ChangeLog 13
}
void MessageLogger::postModuleEndRun(const ModuleDescription& iDescription)
{ unEstablishModule (iDescription, "AfterModEndRun"); }

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

void MessageLogger::preFile()
{  establish("file_open"); }
void MessageLogger::preFileClose( std::string const &, bool )
{  establish("file_close"); }
void MessageLogger::postFile()
{ unEstablish("AfterFile"); }


void
MessageLogger::preEventProcessing( const edm::EventID& iID
                                 , const edm::Timestamp& /*unused*/ )
{
  std::ostringstream ost;
  curr_event_ = iID;
  ost << "Run: " << curr_event_.run() 
      << " Event: " << curr_event_.event();    			// change log 2
  edm::MessageDrop::instance()->runEvent = ost.str();  
  edm::MessageDrop::instance()->setSinglet("PreEventProcessing");// changelog 17
  // Note - module name had not been set here  Similarly in other places where
  // RunEvent carries the new information; we add setSinglet for module name.
}

void
MessageLogger::postEventProcessing(const Event&, const EventSetup&)
{
  edm::MessageDrop::instance()->runEvent = "PostProcessEvent";  
}

void
MessageLogger::preBeginRun( const edm::RunID& iID
                          , const edm::Timestamp& /*unused*/)	// change log 14
{
  std::ostringstream ost;
  ost << "Run: " << iID.run();
  edm::MessageDrop::instance()->runEvent = ost.str();  
  edm::MessageDrop::instance()->setSinglet("PreBeginRun");	// changelog 17
}
void MessageLogger::postBeginRun(const Run&, const EventSetup&)
{ 
  edm::MessageDrop::instance()->runEvent = "PostBeginRun"; 
  edm::MessageDrop::instance()->setSinglet("PostBeginRun");	// changelog 17
  // Note - module name had not been set here
}

void
MessageLogger::prePathBeginRun( const std::string & pathname )	// change log 14
{
  edm::MessageDrop::instance()->setPath( "RPath: ", pathname);	// change log 17
}

void MessageLogger::postPathBeginRun(std::string const&,HLTPathStatus const&)
{ 
  edm::MessageDrop::instance()->setSinglet("PostPathBeginRun");	// changelog 17
}

void
MessageLogger::prePathEndRun( const std::string & pathname )	// change log 14
{
  edm::MessageDrop::instance()->setPath( "RPathEnd: ", pathname);
  								// change log 17
}

void MessageLogger::postPathEndRun(std::string const&,HLTPathStatus const&)
{ 
  edm::MessageDrop::instance()->setSinglet("PostPathEndRun");	// changelog 17
}

void
MessageLogger::prePathBeginLumi( const std::string & pathname )	// change log 14
{
  edm::MessageDrop::instance()->setPath( "LPath: ", pathname);	// change log 17
}

void MessageLogger::postPathBeginLumi(std::string const&,HLTPathStatus const&)
{ 
  edm::MessageDrop::instance()->setSinglet("PostPathBeginLumi"); // changelog 17
}

void
MessageLogger::prePathEndLumi( const std::string & pathname )	// change log 14
{
  edm::MessageDrop::instance()->setPath( "LPathEnd: ", pathname);
  								// change log 17
}

void MessageLogger::postPathEndLumi(std::string const&,HLTPathStatus const&)
{ 
  edm::MessageDrop::instance()->setSinglet("PostPathEndLumi"); // changelog 17
}

void
MessageLogger::preProcessPath( const std::string & pathname )	// change log 14
{
  edm::MessageDrop::instance()->setPath( "PreProcPath ", pathname);
  								// change log 17
}

void MessageLogger::postProcessPath(std::string const&,HLTPathStatus const&)
{ 
  edm::MessageDrop::instance()->setSinglet("PostProcessPath");	// changelog 17
}

void
MessageLogger::preEndRun( const edm::RunID& iID
                        , const edm::Timestamp& /*unused*/)
{
  std::ostringstream ost;
  ost << "End Run: " << iID.run();
  edm::MessageDrop::instance()->runEvent = ost.str();  
  edm::MessageDrop::instance()->setSinglet("PreEndRun");	// changelog 17
}

void MessageLogger::postEndRun(const Run&, const EventSetup&)
{ 
  edm::MessageDrop::instance()->runEvent = "PostEndRun"; 
  edm::MessageDrop::instance()->setSinglet("PostEndRun");	// changelog 17
}

void
MessageLogger::preBeginLumi( const edm::LuminosityBlockID& iID
                          , const edm::Timestamp& /*unused*/)
{
  std::ostringstream ost;
  ost << "Run: " << iID.run() << " Lumi: " << iID.luminosityBlock();
  edm::MessageDrop::instance()->runEvent = ost.str();  
  edm::MessageDrop::instance()->setSinglet("PreBeginLumi");	// changelog 17
}

void MessageLogger::postBeginLumi(const LuminosityBlock&, const EventSetup&)
{ 
  edm::MessageDrop::instance()->runEvent = "PostBeginLumi"; 
  edm::MessageDrop::instance()->setSinglet("PostBeginLumi");	// changelog 17
}

void
MessageLogger::preEndLumi( const edm::LuminosityBlockID& iID
                        , const edm::Timestamp& /*unused*/)
{
  std::ostringstream ost;
  ost << "Run: " << iID.run() << " Lumi: " << iID.luminosityBlock();
  edm::MessageDrop::instance()->runEvent = ost.str();  
  edm::MessageDrop::instance()->setSinglet("PreEndLumi");	// changelog 17
}
void MessageLogger::postEndLumi(const LuminosityBlock&, const EventSetup&)
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
  messageDrop->moduleName = "jobFailure";
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
