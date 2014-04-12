#ifndef PNDiodeTask_H
#define PNDiodeTask_H

#include "DQWorkerTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class PNDiodeTask : public DQWorkerTask {
  public:
    PNDiodeTask();
    ~PNDiodeTask() {}

    bool filterRunType(short const*) override;

    bool analyze(void const*, Collections) override;

    void runOnErrors(EcalElectronicsIdCollection const&, Collections);
    void runOnPnDigis(EcalPnDiodeDigiCollection const&);

  protected:
    bool enable_[ecaldqm::nDCC];
  };

  inline bool PNDiodeTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kMEMTowerIdErrors:
    case kMEMBlockSizeErrors:
    case kMEMChIdErrors:
    case kMEMGainErrors:
      if(_p) runOnErrors(*static_cast<EcalElectronicsIdCollection const*>(_p), _collection);
      return true;
      break;
    case kPnDiodeDigi:
      if(_p) runOnPnDigis(*static_cast<EcalPnDiodeDigiCollection const*>(_p));
      return true;
      break;
    default:
      break;
    }

    return false;
  }

}

#endif

