// CaloConfigHelper.h
//
// Wrapper class for CaloConfig

#ifndef CALO_CONFIG_HELPER_H__
#define CALO_CONFIG_HELPER_H__

#include "CondFormats/L1TObjects/interface/CaloConfig.h"

namespace l1t {

  class CaloConfigHelper {
  public:

    CaloConfigHelper(CaloConfig & db, unsigned fwv, std::string epoch);
    CaloConfigHelper(const CaloConfig & db);
    CaloConfigHelper();
    void UpdatePayload(const CaloConfig * db){ db_ = db; }
    unsigned fwv(){ return db_->uconfig_[0]; }
  private:
    const CaloConfig * db_; // We do not own this pointer...
  };
}

#endif
