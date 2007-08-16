#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {

void LogStatistics() { 
  edm::MessageLoggerQ::MLqSUM ( ); // trigger summary info
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
  edm::MessageLoggerQ::MLqSHT ( ); // Shut the logger up
}

void FlushMessageLog() {
  edm::MessageLoggerQ::MLqFLS ( ); // Flush the message log queue
}

}  // namespace edm
