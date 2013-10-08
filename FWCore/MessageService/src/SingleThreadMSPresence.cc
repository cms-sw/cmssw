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

#include "boost/shared_ptr.hpp"


namespace edm {
namespace service {


SingleThreadMSPresence::SingleThreadMSPresence()
  : Presence()
{
  //std::cout << "SingleThreadMSPresence ctor\n";
  MessageLoggerQ::setMLscribe_ptr(
     boost::shared_ptr<edm::service::AbstractMLscribe> 
     (new ThreadSafeLogMessageLoggerScribe()));
  MessageDrop::instance()->messageLoggerScribeIsRunning = 
  				MLSCRIBE_RUNNING_INDICATOR;
}


SingleThreadMSPresence::~SingleThreadMSPresence()
{
  MessageLoggerQ::MLqEND();
  MessageLoggerQ::setMLscribe_ptr
    (boost::shared_ptr<edm::service::AbstractMLscribe>());
}

} // end of namespace service  
} // end of namespace edm  
