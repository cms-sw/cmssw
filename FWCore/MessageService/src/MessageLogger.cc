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
// $Id: MessageLogger.cc,v 1.11 2006/07/06 14:53:14 lsexton Exp $
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

// system include files
// user include files

#include "FWCore/MessageService/interface/ELcontextSupplier.h"
#include "FWCore/MessageService/interface/MessageLogger.h"

#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"

#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"

#include <sstream>

//#define JMM

#ifdef JMM
#include <iostream>		// JMM debugging
#endif

using namespace edm;
using namespace edm::service;

namespace edm {
namespace service {

bool edm::service::MessageLogger::anyDebugEnabled_   = false;
bool edm::service::MessageLogger::everyDebugEnabled_ = false;

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
  typedef std::vector<std::string>  vString;
   vString  empty_vString;
  
  // grab list of debug-enabled modules
  vString  debugModules;
  try {
    debugModules = 
    	iPS.getUntrackedParameter<vString>("debugModules", empty_vString);
  } catch (...) {
// TODO - Bitch and moan instead of doing what the user might want. 19 Jul. 2006
    debugModules = 
    	iPS.getParameter<vString>("debugModules");
  }

  // grab lists of suppressLEVEL modules
  vString suppressDebug;
  try {
    suppressDebug = 
    	iPS.getUntrackedParameter<vString>("suppressDebug", empty_vString);
  } catch (...) {
// TODO - Bitch and moan instead of doing what the user might want. 19 Jul. 2006
    suppressDebug = 
    	iPS.getParameter<vString>("suppressDebug");
  }

  vString suppressInfo;
  try {
    suppressInfo = 
    	iPS.getUntrackedParameter<vString>("suppressInfo", empty_vString);
  } catch (...) {
// TODO - Bitch and moan instead of doing what the user might want. 19 Jul. 2006
    suppressInfo = 
    	iPS.getParameter<vString>("suppressInfo");
  }

  vString suppressWarning;
  try {
    suppressWarning = 
    	iPS.getUntrackedParameter<vString>("suppressWarning", empty_vString);
  } catch (...) {
// TODO - Bitch and moan instead of doing what the user might want. 19 Jul. 2006
    suppressWarning = 
    	iPS.getParameter<vString>("suppressWarning");
  }

  // Use these lists to prepare a map to use in tracking suppression 

// Do suppressDebug first and suppressWarning last to get proper order
  for( vString::const_iterator it  = suppressDebug.begin();
                               it != suppressDebug.end(); ++it ) {
    suppression_levels_[*it] = ELseverityLevel::ELsev_success;

#ifdef JMM
    std::cout << "suppression_levels_ for module " << *it
              << " set to " << ELseverityLevel::ELsev_success << std::endl; 
#endif
  }
  
  for( vString::const_iterator it  = suppressInfo.begin();
                               it != suppressInfo.end(); ++it ) {
    suppression_levels_[*it] = ELseverityLevel::ELsev_info;

#ifdef JMM
    std::cout << "suppression_levels_ for module " << *it
              << " set to " << ELseverityLevel::ELsev_info << std::endl; 
#endif
  }
  
  for( vString::const_iterator it  = suppressWarning.begin();
                               it != suppressWarning.end(); ++it ) {
    suppression_levels_[*it] = ELseverityLevel::ELsev_warning;

#ifdef JMM
    std::cout << "suppression_levels_ for module " << *it
              << " set to " << ELseverityLevel::ELsev_warning << std::endl; 
#endif
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


#ifdef NEVER
// JMM testing 						17 July 2006
// For lowest order testing, unconditionally set infoEnabled and
// warningEnabled to false
  MessageDrop::instance()->infoEnabled = false;	
  MessageDrop::instance()->warningEnabled = false;	
// End JMM testing 					17 July 2006
#endif

  if ( debugModules.empty()) anyDebugEnabled_ = true;
  for( vString::const_iterator it  = debugModules.begin();
                               it != debugModules.end(); ++it ) {
    if (*it == "*") { 
        everyDebugEnabled_ = true;
      } else {
        debugEnabledModules_.insert(*it); 
      }
  }
  
  MessageLoggerQ::CFG( new ParameterSet(iPS) );

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
  MessageLoggerQ::SUM ( ); // trigger summary info.
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
  curr_module_ = desc.moduleName_;
  curr_module_ += ":";
  curr_module_ += desc.moduleLabel_;
  MessageDrop::instance()->moduleName = curr_module_ + "{*ctor*}";  
  if (!anyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = false;
  } else if (everyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = true;
  } else {
    MessageDrop::instance()->debugEnabled = 
    			debugEnabledModules_.count(desc.moduleLabel_);
  }
}

void
MessageLogger::preSource()
{
  curr_module_ = "main_input";
  curr_module_ += ":";
  curr_module_ += "source";
  MessageDrop::instance()->moduleName = curr_module_;  
  if (!anyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = false;
  } else if (everyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = true;
  } else {
    MessageDrop::instance()->debugEnabled = 
    			debugEnabledModules_.count("source");
  }
}


void
MessageLogger::preModuleConstruction(const ModuleDescription& desc)
{
  // LogInfo("preModule") << "Module:" << desc.moduleLabel();
  curr_module_ = desc.moduleName_;
  curr_module_ += ":";
  curr_module_ += desc.moduleLabel_;
  MessageDrop::instance()->moduleName = curr_module_ + "{ctor}";  
  if (!anyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = false;
  } else if (everyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = true;
  } else {
    MessageDrop::instance()->debugEnabled = 
    			debugEnabledModules_.count(desc.moduleLabel_);
  }
}

void
MessageLogger::preModule(const ModuleDescription& desc)
{
  // LogInfo("preModule") << "Module:" << desc.moduleLabel();
  curr_module_ = desc.moduleName_;
  curr_module_ += ":";
  curr_module_ += desc.moduleLabel_;
  MessageDrop::instance()->moduleName = curr_module_;  
  if (!anyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = false;
  } else if (everyDebugEnabled_) {
    MessageDrop::instance()->debugEnabled = true;
  } else {
    MessageDrop::instance()->debugEnabled = 
    			debugEnabledModules_.count(desc.moduleLabel_);
  }

#ifdef JMM
  std::cout << "Searching map for " << desc.moduleLabel_ << std::endl;
#endif

  std::map<const std::string,ELseverityLevel>::const_iterator it =
       suppression_levels_.find(desc.moduleLabel_);
  if ( it != suppression_levels_.end() ) {
    MessageDrop::instance()->debugEnabled  = MessageDrop::instance()->debugEnabled 
                                           && (it->second < ELseverityLevel::ELsev_success );
    MessageDrop::instance()->infoEnabled    = (it->second < ELseverityLevel::ELsev_info );
    MessageDrop::instance()->warningEnabled = (it->second < ELseverityLevel::ELsev_warning );
  } else {
    MessageDrop::instance()->infoEnabled    = true;
    MessageDrop::instance()->warningEnabled = true;
  }
#ifdef JMM
  std::cout << "Setting MessageDrop::debugEnabled to "
            << MessageDrop::instance()->debugEnabled << std::endl;
  std::cout << "Setting MessageDrop::infoEnabled to "
            << MessageDrop::instance()->infoEnabled << std::endl;
  std::cout << "Setting MessageDrop::warningEnabled to " 
            << MessageDrop::instance()->warningEnabled << std::endl;
#endif
}

void
MessageLogger::postSourceConstruction(const ModuleDescription& iDescription)
{
  // LogInfo("postModule") << "Module:" << iDescription.moduleLabel_
  //                      << " finished";
  curr_module_ = "AfterSourceConstruction";
  MessageDrop::instance()->moduleName = curr_module_;  
}

void
MessageLogger::postModuleConstruction(const ModuleDescription& iDescription)
{
  // LogInfo("postModule") << "Module:" << iDescription.moduleLabel_
  //                      << " finished";
  curr_module_ = "AfterModuleConstruction";
  MessageDrop::instance()->moduleName = curr_module_;  
}

void
MessageLogger::postSource()
{
  // LogInfo("postModule") << "Module:" << iDescription.moduleLabel_
  //                      << " finished";
  curr_module_ = "PostSource";
  MessageDrop::instance()->moduleName = curr_module_;  
}

void
MessageLogger::postModule(const ModuleDescription& iDescription)
{
  // LogInfo("postModule") << "Module:" << iDescription.moduleLabel_
  //                      << " finished";
  curr_module_ = "PostModule";
  MessageDrop::instance()->moduleName = curr_module_;  
}

} // end of namespace service  
} // end of namespace edm  
