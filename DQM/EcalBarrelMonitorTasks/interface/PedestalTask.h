#ifndef PedestalTask_H
#define PedestalTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class PedestalTask : public DQWorkerTask {
  public:
    PedestalTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PedestalTask() {}

    bool filterRunType(const std::vector<short>&);

    void analyze(const void*, Collections);

    void runOnDigis(const EcalDigiCollection&);
    void runOnPnDigis(const EcalPnDiodeDigiCollection&);

  protected:
    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;

    bool enable_[BinService::nDCC];
  };

  inline void PedestalTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEBDigi:
    case kEEDigi:
      runOnDigis(*static_cast<const EcalDigiCollection*>(_p));
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
