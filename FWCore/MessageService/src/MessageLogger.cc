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
// $Id: MessageLogger.cc,v 1.31 2009/07/08 20:26:38 fischler Exp $
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
#include <string>
#include <map>

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
	: curr_module_("BeginningJob")
        , debugEnabled_(false)
	, messageServicePSetHasBeenValidated_(false)
	, messageServicePSetValidatationResults_() 
	, nonModule_debugEnabled(false)
	, nonModule_infoEnabled(true)
	, nonModule_warningEnabled(true)
{
  // prepare cfg validation string for later use
  MessageServicePSetValidation validator;
  messageServicePSetValidatationResults_ = validator(iPS);	// change log 12
  
  typedef std::vector<std::string>  vString;
  vString  empty_vString;
  vString  debugModules;
  vString suppressDebug;
  vString suppressWarning;
  vString suppressInfo;

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
  } catch (cms::Exception& e) {					// change log 13
  }
  
  // Use these lists to prepare a map to use in tracking suppression 

  // Do suppressDebug first and suppressWarning last to get proper order
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
			       std::string const & whichPhase)	// ChangeLog 13
{
  MessageDrop* messageDrop = MessageDrop::instance();
  nonModule_debugEnabled   = messageDrop->debugEnabled;
  nonModule_infoEnabled    = messageDrop->infoEnabled;
  nonModule_warningEnabled = messageDrop->warningEnabled;

  //cache the value to improve performance based on profiling studies
  std::map<const ModuleDescription*,std::string>::const_iterator itFind = descToCalcName_.find(&desc);
  if ( itFind == descToCalcName_.end()) {
    curr_module_ = desc.moduleName();
    curr_module_ += ":";
    curr_module_ += desc.moduleLabel();
    //cache this value to improve performance based on profiling studies
    descToCalcName_[&desc]=curr_module_;
    messageDrop->moduleName = curr_module_ + whichPhase;  
  } else {
    messageDrop->moduleName = itFind->second + whichPhase;
  }
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
  } else {
    messageDrop->infoEnabled    = true;
    messageDrop->warningEnabled = true;
  }
} // establishModule

void
MessageLogger::unEstablishModule(ModuleDescription const & desc, 
			         std::string const & state)
{
  MessageDrop* messageDrop = MessageDrop::instance();
  messageDrop->moduleName = state;
  messageDrop->debugEnabled   = nonModule_debugEnabled;
  messageDrop->infoEnabled    = nonModule_infoEnabled;
  messageDrop->warningEnabled = nonModule_warningEnabled;
}

void
MessageLogger::establish(std::string const & state)
{
  MessageDrop* messageDrop = MessageDrop::instance();
  curr_module_ = state;
  messageDrop->moduleName = curr_module_;  
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
  } else {
    messageDrop->infoEnabled    = true;
    messageDrop->warningEnabled = true;
  }
}

void
MessageLogger::unEstablish(std::string const & state)
{
  MessageDrop::instance()->moduleName = state;  
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
  establishModule (desc,"@ctor");				// ChangeLog 13
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
  establishModule (desc,"@sourceConstruction");			// ChangeLog 13
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
  MessageDrop::instance()->moduleName = "AfterBeginJob";  
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
void MessageLogger::preFileClose()
{  establish("file_close"); }
void MessageLogger::postFile()
{ unEstablish("AfterFile"); }


void
MessageLogger::preEventProcessing( const edm::EventID& iID
                                 , const edm::Timestamp& iTime )
{
  std::ostringstream ost;
  curr_event_ = iID;
  ost << "Run: " << curr_event_.run() 
      << " Event: " << curr_event_.event();    			// change log 2
  edm::MessageDrop::instance()->runEvent = ost.str();  
}

void
MessageLogger::postEventProcessing(const Event&, const EventSetup&)
{
  edm::MessageDrop::instance()->runEvent = "PostProcessEvent";  
}

void
MessageLogger::preBeginRun( const edm::RunID& iID
                          , const edm::Timestamp& iTime )	// change log 14
{
  std::ostringstream ost;
  ost << "Run: " << iID.run();
  edm::MessageDrop::instance()->runEvent = ost.str();  
}
void MessageLogger::postBeginRun(const Run&, const EventSetup&)
{ edm::MessageDrop::instance()->runEvent = "PostBeginRun"; }

void
MessageLogger::prePathBeginRun( const std::string & pathname )	// change log 14
{
  std::ostringstream ost;
  ost << "RPath: " << pathname;
  edm::MessageDrop::instance()->moduleName = ost.str();  
}
void MessageLogger::postPathBeginRun(std::string const&,HLTPathStatus const&)
{ edm::MessageDrop::instance()->moduleName = "PostPathBeginRun"; }

void
MessageLogger::prePathEndRun( const std::string & pathname )	// change log 14
{
  std::ostringstream ost;
  ost << "RPathEnd: " << pathname;
  edm::MessageDrop::instance()->moduleName = ost.str();  
}
void MessageLogger::postPathEndRun(std::string const&,HLTPathStatus const&)
{ edm::MessageDrop::instance()->moduleName = "PostPathEndRun"; }

void
MessageLogger::prePathBeginLumi( const std::string & pathname )	// change log 14
{
  std::ostringstream ost;
  ost << "LPath: " << pathname;
  edm::MessageDrop::instance()->moduleName = ost.str();  
}
void MessageLogger::postPathBeginLumi(std::string const&,HLTPathStatus const&)
{ edm::MessageDrop::instance()->moduleName = "PostPathBeginLumi"; }

void
MessageLogger::prePathEndLumi( const std::string & pathname )	// change log 14
{
  std::ostringstream ost;
  ost << "LPathEnd: " << pathname;
  edm::MessageDrop::instance()->moduleName = ost.str();  
}
void MessageLogger::postPathEndLumi(std::string const&,HLTPathStatus const&)
{ edm::MessageDrop::instance()->moduleName = "PostPathEndLumi"; }

void
MessageLogger::preProcessPath( const std::string & pathname )	// change log 14
{
  std::ostringstream ost;
  ost << "PreProcPath " << pathname;
  edm::MessageDrop::instance()->moduleName = ost.str();  
}
void MessageLogger::postProcessPath(std::string const&,HLTPathStatus const&)
{ edm::MessageDrop::instance()->moduleName = "PostProcessPath"; }

void
MessageLogger::preEndRun( const edm::RunID& iID
                        , const edm::Timestamp& iTime )
{
  std::ostringstream ost;
  ost << "End Run: " << iID.run();
  edm::MessageDrop::instance()->runEvent = ost.str();  
}
void MessageLogger::postEndRun(const Run&, const EventSetup&)
{ edm::MessageDrop::instance()->runEvent = "PostEndRun"; }

void
MessageLogger::preBeginLumi( const edm::LuminosityBlockID& iID
                          , const edm::Timestamp& iTime )
{
  std::ostringstream ost;
  ost << "Run: " << iID.run() << " Lumi: " << iID.luminosityBlock();
  edm::MessageDrop::instance()->runEvent = ost.str();  
}
void MessageLogger::postBeginLumi(const LuminosityBlock&, const EventSetup&)
{ edm::MessageDrop::instance()->runEvent = "PostBeginLumi"; }

void
MessageLogger::preEndLumi( const edm::LuminosityBlockID& iID
                        , const edm::Timestamp& iTime )
{
  std::ostringstream ost;
  ost << "Run: " << iID.run() << " Lumi: " << iID.luminosityBlock();
  edm::MessageDrop::instance()->runEvent = ost.str();  
}
void MessageLogger::postEndLumi(const LuminosityBlock&, const EventSetup&)
{ edm::MessageDrop::instance()->runEvent = "PostEndLumi"; }

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
