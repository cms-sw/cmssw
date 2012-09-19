#ifndef EcalDQMClientUtils_H
#define EcalDQMClientUtils_H

#include "DQM/EcalCommon/interface/EcalDQMBinningService.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"

class DetId;
namespace edm {
  class FileInPath;
}

namespace ecaldqm
{
  typedef EcalDQMBinningService BinService;

  bool applyMask(BinService::BinningType, DetId const&, uint32_t);

  void setStatuses(EcalDQMChannelStatus const*, EcalDQMTowerStatus const*);
  void readPNMaskMap(edm::FileInPath const&);
}

#endif
