// ----------------------------------------------------------------------
//
// SingleThreadMSPresence.cc
//
// Changes:
//
// 

#include "FWCore/MessageService/interface/SingleThreadMSPresence.h"
#include "FWCore/MessageService/interface/ThreadSafeLogMessageLoggerScribe.h"

#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include <memory>


namespace edm {
namespace service {


SingleThreadMSPresence::SingleThreadMSPresence()
  : Presence()
{
  //std::cout << "SingleThreadMSPresence ctor\n";
  MessageLoggerQ::setMLscribe_ptr(std::shared_ptr<edm::service::AbstractMLscribe>(std::make_shared<ThreadSafeLogMessageLoggerScribe>()));
  MessageDrop::instance()->messageLoggerScribeIsRunning = 
  				MLSCRIBE_RUNNING_INDICATOR;
}


SingleThreadMSPresence::~SingleThreadMSPresence()
{
  MessageLoggerQ::MLqEND();
  MessageLoggerQ::setMLscribe_ptr
    (std::shared_ptr<edm::service::AbstractMLscribe>());
}

} // end of namespace service  
} // end of namespace edm  
