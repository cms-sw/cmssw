#include <vector>
#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace FWCore_MessageLogger {
  struct dictionary {
    std::vector<edm::ErrorSummaryEntry> w_v_es;
  };
}
