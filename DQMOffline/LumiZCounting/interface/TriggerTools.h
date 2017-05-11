#ifndef DQMOFFLINE_LUMIZCOUNTING_TRIGGERTOOLS_H
#define DQMOFFLINE_LUMIZCOUNTING_TRIGGERTOOLS_H

#include "DQMOffline/LumiZCounting/interface/MiniBaconDefs.h"
#include "DQMOffline/LumiZCounting/interface/TriggerRecord.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include <vector>

namespace baconhep {

class TriggerTools
{
public:
  static TriggerObjects  matchHLT(const double eta, const double phi, 
				  const std::vector<TriggerRecord> &triggerRecords,
				  const trigger::TriggerEvent &triggerEvent);
};

}
#endif
