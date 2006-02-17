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
// $Id: MessageLogger.cc,v 1.2 2006/02/17 16:05:27 jbk Exp $
//

// system include files

// user include files

#include "FWCore/MessageService/interface/ELcontextSupplier.h"
#include "FWCore/MessageService/interface/MessageLogger.h"

#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"

#include <sstream>

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
    debugModules = 
    	iPS.getParameter<vString>("debugModules");
  }
  // set up for tracking whether current module is debug-enabled
  if (!debugModules.empty()) anyDebugEnabled_ = true;
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

  iRegistry.watchPreProcessEvent(this,&MessageLogger::preEventProcessing);
  iRegistry.watchPostProcessEvent(this,&MessageLogger::postEventProcessing);

  iRegistry.watchPreModule(this,&MessageLogger::preModule);
  iRegistry.watchPostModule(this,&MessageLogger::postModule);

  
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
  ost << curr_event_.run() << "/" << curr_event_.event();
  edm::MessageDrop::instance()->runEvent = ost.str();  
}
void
MessageLogger::postEventProcessing(const Event&, const EventSetup&)
{
  MessageDrop::instance()->runEvent = "BetweenEvents";  
}

void
MessageLogger::preModule(const ModuleDescription& desc)
{
  // LogInfo("preModule") << "Module:" << desc.moduleLabel_;
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
}

void
MessageLogger::postModule(const ModuleDescription& iDescription)
{
  // LogInfo("postModule") << "Module:" << iDescription.moduleLabel_
  //                      << " finished";
  curr_module_ = "BetweenModules";
  MessageDrop::instance()->moduleName = curr_module_;  
}

} // end of namespace service  
} // end of namespace edm  
