#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include <cstring>

namespace edm {
void LogStatistics() { 
  edm::MessageLoggerQ::SUM ( ); // trigger summary info
}

}

