// CaloConfigHelper.cc

#include "L1Trigger/L1TCalorimeter/interface/CaloConfigHelper.h"

namespace l1t {
  
  CaloConfigHelper::CaloConfigHelper(CaloConfig & db, unsigned fwv_layer2, unsigned fedid_layer2, std::string epoch) : db_(&db) {
    db.uconfig_.push_back(fwv_layer2);
    db.uconfig_.push_back(fedid_layer2);
    db.sconfig_.push_back(epoch);
  }
  CaloConfigHelper::CaloConfigHelper(const CaloConfig & db) : db_(&db) {
    
  }
  CaloConfigHelper::CaloConfigHelper() {
    static CaloConfig db;
    db_ = &db;
  }
}
