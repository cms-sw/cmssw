#include <vector>
#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"

namespace FWCore_MessageLogger {
  struct dictionary {
    std::vector<edm::ErrorSummaryEntry> w_v_es;
  };
}
