#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"
#include "L1Trigger/L1TGlobal/interface/TriggerMenu.h"

namespace L1Trigger_L1TGlobal {
  struct dictionary {
    std::vector<MuonTemplate> dummy1;
    std::vector<CaloTemplate> dummy2;
    std::vector<CorrelationTemplate> dummy3;
    std::vector<CorrelationThreeBodyTemplate> dummy4;
    std::vector<CorrelationWithOverlapRemovalTemplate> dummy5;
    std::vector<MuonShowerTemplate> dummy6;
    std::vector<AXOL1TLTemplate> dummy7;
  };
}  // namespace L1Trigger_L1TGlobal
