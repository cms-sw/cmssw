#ifndef PedestalTask_H
#define PedestalTask_H

#include "DQWorkerTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class PedestalTask : public DQWorkerTask {
  public:
    PedestalTask();
    ~PedestalTask() {}

    bool filterRunType(short const*) override;

    bool analyze(void const*, Collections) override;

    template<typename DigiCollection> void runOnDigis(DigiCollection const&);
    void runOnPnDigis(EcalPnDiodeDigiCollection const&);

  private:
    void setParams(edm::ParameterSet const&) override;

    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;

    bool enable_[nDCC];
  };

  inline bool PedestalTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kEBDigi:
      if(_p) runOnDigis(*static_cast<EBDigiCollection const*>(_p));
      return true;
      break;
    case kEEDigi:
      if(_p) runOnDigis(*static_cast<EEDigiCollection const*>(_p));
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
