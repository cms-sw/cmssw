// CaloConfigHelper.cc

#include "L1Trigger/L1TCalorimeter/interface/CaloConfigHelper.h"

namespace l1t {
  
  CaloConfigHelper::CaloConfigHelper(CaloConfig & db, unsigned fwv, std::string epoch) : db_(&db) {
    db.uconfig_.push_back(fwv);
    db.sconfig_.push_back(epoch);
  }
  CaloConfigHelper::CaloConfigHelper(const CaloConfig & db) : db_(&db) {
    
  }
  CaloConfigHelper::CaloConfigHelper() {
    static CaloConfig db;
    db_ = &db;
  }
}
