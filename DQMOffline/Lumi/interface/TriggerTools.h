#ifndef DQMOFFLINE_LUMI_TRIGGERTOOLS_H
#define DQMOFFLINE_LUMI_TRIGGERTOOLS_H

#include "DQMOffline/Lumi/interface/TriggerDefs.h"
#include "DQMOffline/Lumi/interface/TriggerRecord.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include <vector>

namespace ZCountingTrigger {

class TriggerTools
{
public:
  static TriggerObjects  matchHLT(const double eta, const double phi, 
				  const std::vector<TriggerRecord> &triggerRecords,
				  const trigger::TriggerEvent &triggerEvent);
};

}
#endif
