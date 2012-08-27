#ifndef PNIntegrityTask_H
#define PNIntegrityTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

namespace ecaldqm {

  class PNIntegrityTask : public DQWorkerTask {
  public:
    PNIntegrityTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PNIntegrityTask();

    void analyze(const void*, Collections);

    void runOnErrors(const EcalElectronicsIdCollection &, Collections);

    enum MESets {
      kMEMChId,
      kMEMGain,
      kMEMBlockSize,
      kMEMTowerId,
      nMESets
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);
  };

  inline void PNIntegrityTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kMEMTowerIdErrors:
    case kMEMBlockSizeErrors:
    case kMEMChIdErrors:
    case kMEMGainErrors:
      runOnErrors(*static_cast<const EcalElectronicsIdCollection*>(_p), _collection);
      break;
    default:
      break;
    }
  }

}

#endif

