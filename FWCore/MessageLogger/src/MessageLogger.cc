#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include <cstring>

namespace edm {

void LogStatistics() { 
  edm::MessageLoggerQ::SUM ( ); // trigger summary info
}

LogDebug_ dummyLogDebugObject_( "dummy_id", __FILE__, __LINE__ );
LogTrace_ dummyLogTraceObject_( "dummy_id" );

}  // namespace edm
