#ifndef EcalDQMClientUtils_H
#define EcalDQMClientUtils_H

#include "DQM/EcalCommon/interface/EcalDQMBinningService.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"

class DetId;

namespace ecaldqm
{
  typedef EcalDQMBinningService BinService;

  float maskQuality(BinService::BinningType, DetId const&, uint32_t, int);

  void setStatuses(EcalDQMChannelStatus const*, EcalDQMTowerStatus const*);
}

#endif
