#ifndef L1TMuonEndCap_ConditionHelper_hh
#define L1TMuonEndCap_ConditionHelper_hh

#include "FWCore/Framework/interface/ESHandle.h"

// forwards
namespace edm {
  class EventSetup;
}

class L1TMuonEndCapParams;
class L1TMuonEndCapForest;
class PtAssignmentEngine;


// class declaration
class ConditionHelper {
public:
  ConditionHelper();
  ~ConditionHelper();

  void checkAndUpdateConditions(const edm::EventSetup& iSetup, PtAssignmentEngine& pt_assign_engine_);

  const L1TMuonEndCapParams& getParams() const { return *params_; }
  const L1TMuonEndCapForest& getForest() const { return *forest_; }

private:
  unsigned long long params_cache_id_;
  unsigned long long forest_cache_id_;

  edm::ESHandle<L1TMuonEndCapParams> params_;
  edm::ESHandle<L1TMuonEndCapForest> forest_;
};


#endif
