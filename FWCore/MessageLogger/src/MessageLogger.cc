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
// $Id: MessageLogger.cc,v 1.10 2006/01/18 00:36:09 fischler Exp $
//

// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/ELcontextSupplier.h"

#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/EDProduct/interface/EventID.h"
#include "FWCore/EDProduct/interface/Timestamp.h"

#include <sstream>

using namespace edm::service;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

bool MessageLogger::anyDebugEnabled_   = false;
bool MessageLogger::everyDebugEnabled_ = false;

//
// constructors and destructor
//
MessageLogger::MessageLogger( ParameterSet const & iPS
                            , ActivityRegistry   & iRegistry
                            )
			    : curr_module_("BeginningJob"),
			      debugEnabled_(false)
{
  typedef std::vector<std::string>  vString;
   vString  empty_vString;
  
  // grab list of debug-enabled modules
  vString  debugModules
     = iPS.getUntrackedParameter<vString>("debugModules", empty_vString);
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

// MessageLogger::MessageLogger(const MessageLogger& rhs)
// {
//    // do actual copying here;
// }

//MessageLogger::~MessageLogger()
//{
//}

//
// assignment operators
//
// const MessageLogger& MessageLogger::operator=(const MessageLogger& rhs)
// {
//   //An exception safe implementation is
//   MessageLogger temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// free funtions
//

void edm::LogStatistics() {
  edm::MessageLoggerQ::SUM ( ); // trigger summary info.
}


//
// member functions
//

void MessageLogger::fillErrorObj(edm::ErrorObj& obj) const
{
  obj.setModule(curr_module_);
  std::ostringstream ost;
  ost << curr_event_.run() << "/" << curr_event_.event();
  obj.setContext(ost.str());
}

void
MessageLogger::postBeginJob()
{
  edm::LogInfo("postBeginJob") << "Job started";

  curr_module_ = "BetweenModules";
  curr_event_ = EventID();
}
void
MessageLogger::postEndJob()
{
  edm::LogInfo("postEndJob") << "Job ended";
  MessageLoggerQ::SUM ( ); // trigger summary info.
}

void
MessageLogger::preEventProcessing( const edm::EventID& iID
                                 , const edm::Timestamp& iTime
                                 )
{
  edm::LogInfo("preEventProcessing") << "Processing event: " << iID
                                     << " time: " << iTime.value();

  curr_event_ = iID;
}
void
MessageLogger::postEventProcessing(const Event&, const EventSetup&)
{
  edm::LogInfo("postEventProcessing") << "Finished event";

  curr_event_ = EventID();
}

void
MessageLogger::preModule(const ModuleDescription& desc)
{
  LogInfo("preModule") << "Module:" << desc.moduleLabel_;
  curr_module_ = desc.moduleName_;
  curr_module_ += ":";
  curr_module_ += desc.moduleLabel_;
  if (!anyDebugEnabled_) {
    debugEnabled_ = false;
  } else if (everyDebugEnabled_) {
    debugEnabled_ = true;
  } else {
    debugEnabled_ = debugEnabledModules_.count(desc.moduleLabel_);
  }
}

void
MessageLogger::postModule(const ModuleDescription& iDescription)
{
  LogInfo("postModule") << "Module:" << iDescription.moduleLabel_
                        << " finished";
  curr_module_ = "BetweenModules";
}

//
// const member functions
//

//
// static member functions
//
