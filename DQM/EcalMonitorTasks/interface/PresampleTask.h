#ifndef PresampleTask_H
#define PresampleTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

namespace ecaldqm
{
  class PresampleTask : public DQWorkerTask {
  public:
    PresampleTask();
    ~PresampleTask() {}

    bool filterRunType(short const*) override;

    bool analyze(void const*, Collections) override;

    template<typename DigiCollection> void runOnDigis(DigiCollection const&);

  private:
    void setParams(edm::ParameterSet const&) override;

    int pulseMaxPosition_;
    int nSamples_;
  };

  inline bool PresampleTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kEBDigi:
      if(_p) runOnDigis(*static_cast<EBDigiCollection const*>(_p));
      return true;
    case kEEDigi:
      if(_p) runOnDigis(*static_cast<EEDigiCollection const*>(_p));
      return true;
      break;
    default:
      break;
    }

    return false;
  }
}

#endif

