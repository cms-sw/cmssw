#ifndef PNDiodeTask_H
#define PNDiodeTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class PNDiodeTask : public DQWorkerTask {
  public:
    PNDiodeTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PNDiodeTask() {}

    bool filterRunType(const std::vector<short>&);

    void analyze(const void*, Collections);

    void runOnErrors(const EcalElectronicsIdCollection &, Collections);
    void runOnPnDigis(const EcalPnDiodeDigiCollection&);

  protected:
    bool enable_[BinService::nDCC];
  };

  inline void PNDiodeTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kMEMTowerIdErrors:
    case kMEMBlockSizeErrors:
    case kMEMChIdErrors:
    case kMEMGainErrors:
      runOnErrors(*static_cast<const EcalElectronicsIdCollection*>(_p), _collection);
      break;
    case kPnDiodeDigi:
      runOnPnDigis(*static_cast<const EcalPnDiodeDigiCollection*>(_p));
      break;
    default:
      break;
    }
  }

}

#endif

