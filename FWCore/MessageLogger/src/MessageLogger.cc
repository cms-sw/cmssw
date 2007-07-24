#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"

namespace edm {

void LogStatistics() { 
  edm::MessageLoggerQ::SUM ( ); // trigger summary info
}

LogDebug_ dummyLogDebugObject_( "dummy_id", __FILE__, __LINE__ );
LogTrace_ dummyLogTraceObject_( "dummy_id" );

bool isDebugEnabled() {
  return ( edm::MessageDrop::instance()->debugEnabled );
}

bool isInfoEnabled() {
  return( edm::MessageDrop::instance()->infoEnabled );
}

bool isWarningEnabled() {
  return( edm::MessageDrop::instance()->warningEnabled );
}

void HaltMessageLogging() {
  edm::MessageLoggerQ::SHT ( ); // Shut the logger up
}

}  // namespace edm
