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
// $Id: MessageLogger.cc,v 1.27 2008/06/24 20:31:40 fischler Exp $
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
// 3 mf   6/27/06	PreModuleCOnstruction and PreSourceCOnstruction get
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

// system include files
// user include files

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageService/interface/MessageLogger.h"

#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <sstream>
#include <string>
#include <map>

static const std::string kPostModule("PostModule");
static const std::string kSource("main_input:source");

using namespace edm;
using namespace edm::service;

namespace edm {
namespace service {

bool edm::service::MessageLogger::anyDebugEnabled_     = false;
bool edm::service::MessageLogger::everyDebugEnabled_   = false;
bool edm::service::MessageLogger::fjrSummaryRequested_ = false;
  
//
// constructors and destructor
//
edm::service::MessageLogger::
MessageLogger( ParameterSet const & iPS
             , ActivityRegistry   & iRegistry
                            )
	: curr_module_("BeginningJob")
        , debugEnabled_(false)
{
  // decide whether a summary should be placed in job report
  fjrSummaryRequested_ = 
    	iPS.getUntrackedParameter<bool>("messageSummaryToJobReport", false);

  typedef std::vector<std::string>  vString;
   vString  empty_vString;
  
  // grab list of debug-enabled modules
  vString  debugModules;
  debugModules = 
    	iPS.getUntrackedParameter<vString>("debugModules", empty_vString);

  // grab lists of suppressLEVEL modules
  vString suppressDebug;
  suppressDebug = 
    	iPS.getUntrackedParameter<vString>("suppressDebug", empty_vString);

  vString suppressInfo;
  suppressInfo = 
    	iPS.getUntrackedParameter<vString>("suppressInfo", empty_vString);

  vString suppressWarning;
  suppressWarning = 
    	iPS.getUntrackedParameter<vString>("suppressWarning", empty_vString);

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
    MessageDrop::instance()->debugEnabled = false;	// change log 2
  } else {
    anyDebugEnabled_ = true;
    MessageDrop::instance()->debugEnabled = true;
    // this will be over-ridden when specific modules are entered
  }

  if ( debugModules.empty()) anyDebugEnabled_ = true;
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
    MessageLoggerQ::MLqJOB( jr_name_p ); 			// change log 8
  }
  
  								// change log 7
  std::string jm = edm::MessageDrop::instance()->jobMode; 
  std::string * jm_p = new std::string(jm);
  MessageLoggerQ::MLqMOD( jm_p ); 				// change log 8
  
  MessageLoggerQ::MLqCFG( new ParameterSet(iPS) );		// change log 8

  iRegistry.watchPostBeginJob(this,&MessageLogger::postBeginJob);
  iRegistry.watchPostEndJob(this,&MessageLogger::postEndJob);

  iRegistry.watchPreModuleConstruction(this,&MessageLogger::preModuleConstruction);
  iRegistry.watchPostModuleConstruction(this,&MessageLogger::postModuleConstruction);
								// change log 3

  iRegistry.watchPreSourceConstruction(this,&MessageLogger::preSourceConstruction);
  iRegistry.watchPostSourceConstruction(this,&MessageLogger::postSourceConstruction);
								// change log 3

  iRegistry.watchPreProcessEvent(this,&MessageLogger::preEventProcessing);
  iRegistry.watchPostProcessEvent(this,&MessageLogger::postEventProcessing);

  iRegistry.watchPreModule(this,&MessageLogger::preModule);
  iRegistry.watchPostModule(this,&MessageLogger::postModule);

  iRegistry.watchPreSource(this,&MessageLogger::preSource);
  iRegistry.watchPostSource(this,&MessageLogger::postSource);
								// change log 3
}

// edm::service::
// MessageLogger::MessageLogger(const MessageLogger& rhs)
// {
//    // do actual copying here;
// }

// edm::service::
// MessageLogger::~MessageLogger()
//{
//}

//
// assignment operators
//
// const // edm::service::MessageLogger& MessageLogger::operator=(const MessageLogger& rhs)
// {
//   //An exception safe implementation is
//   MessageLogger temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void
MessageLogger::postBeginJob()
{
  MessageDrop::instance()->runEvent = "BeforeEvents";  
  MessageDrop::instance()->moduleName = "";  
}

void
MessageLogger::postEndJob()
{
  SummarizeInJobReport();     // Put summary info into Job Rep  // change log 10
  MessageLoggerQ::MLqSUM ( ); // trigger summary info.		// change log 8
}

void
MessageLogger::preEventProcessing( const edm::EventID& iID
                                 , const edm::Timestamp& iTime
                                 )
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
  // MessageDrop::instance()->runEvent = "BetweenEvents";  	// change log 4
}

void
MessageLogger::preSourceConstruction(const ModuleDescription& desc)
{
  curr_module_ = desc.moduleName();
  curr_module_ += ":";
  curr_module_ += desc.moduleLabel();
  MessageDrop::instance()->moduleName = curr_module_ + "{*ctor*}";  
  if (!anyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = false;
  } else if (everyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = true;
  } else {
    MessageDrop::instance()->debugEnabled = 
    			debugEnabledModules_.count(desc.moduleLabel());
  }
}

void
MessageLogger::preSource()
{
  MessageDrop* messageDrop = MessageDrop::instance();
  curr_module_ = kSource;
  //curr_module_ += ":";
  //curr_module_ += "source";
  messageDrop->moduleName = curr_module_;  
  if (!anyDebugEnabled_) {
    messageDrop->debugEnabled = false;
  } else if (everyDebugEnabled_) {
    messageDrop->debugEnabled = true;
  } else {
    messageDrop->debugEnabled = 
    		debugEnabledModules_.count("source");
  }
  std::map<const std::string,ELseverityLevel>::const_iterator it =
       suppression_levels_.find("source");
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
MessageLogger::preModuleConstruction(const ModuleDescription& desc)
{
  // LogInfo("preModule") << "Module:" << desc.moduleLabel();
  curr_module_ = desc.moduleName();
  curr_module_ += ":";
  curr_module_ += desc.moduleLabel();
  
  MessageDrop::instance()->moduleName = curr_module_ + "{ctor}";  
  if (!anyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = false;
  } else if (everyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = true;
  } else {
    MessageDrop::instance()->debugEnabled = 
    			debugEnabledModules_.count(desc.moduleLabel());
  }
}

void
MessageLogger::preModule(const ModuleDescription& desc)
{
  // LogInfo("preModule") << "Module:" << desc.moduleLabel();
  //cache the value to improve performance based on profiling studies
  MessageDrop* messageDrop = MessageDrop::instance();
  std::map<const ModuleDescription*,std::string>::const_iterator itFind = descToCalcName_.find(&desc);
  if ( itFind == descToCalcName_.end()) {
    curr_module_ = desc.moduleName();
    curr_module_ += ":";
    curr_module_ += desc.moduleLabel();
    //cache this value to improve performance based on profiling studies
    descToCalcName_[&desc]=curr_module_;
    messageDrop->moduleName = curr_module_;  
  } else {
    messageDrop->moduleName = itFind->second;
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
}

void
MessageLogger::postSourceConstruction(const ModuleDescription& iDescription)
{
  // LogInfo("postModule") << "Module:" << iDescription.moduleLabel();
  //                      << " finished";
  curr_module_ = "AfterSourceConstruction";
  MessageDrop::instance()->moduleName = curr_module_;  
}

void
MessageLogger::postModuleConstruction(const ModuleDescription& iDescription)
{
  // LogInfo("postModule") << "Module:" << iDescription.moduleLabel()
  //                      << " finished";
  curr_module_ = "AfterModuleConstruction";
  MessageDrop::instance()->moduleName = curr_module_;  
}

void
MessageLogger::postSource()
{
  // LogInfo("postModule") << "Module:" << iDescription.moduleLabel()
  //                      << " finished";
  curr_module_ = "PostSource";
  MessageDrop::instance()->moduleName = curr_module_;  
}

void
MessageLogger::postModule(const ModuleDescription& iDescription)
{
  // LogInfo("postModule") << "Module:" << iDescription.moduleLabel()
  //                      << " finished";
  //curr_module_ = kPostModule;
  //MessageDrop::instance()->moduleName = curr_module_;  
  MessageDrop::instance()->moduleName = kPostModule;
}

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
