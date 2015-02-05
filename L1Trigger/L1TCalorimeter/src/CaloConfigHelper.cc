// CaloConfigHelper.cc

#include "L1Trigger/L1TCalorimeter/interface/CaloConfigHelper.h"

namespace l1t {
  
  CaloConfigHelper::CaloConfigHelper(CaloConfig & db, unsigned fwv, std::string epoch) : db_(db) {
    db_.uconfig_.push_back(fwv);
    db_.sconfig_.push_back(epoch);
  }
  CaloConfigHelper::CaloConfigHelper(CaloConfig & db) : db_(db) {
    
  }
}
