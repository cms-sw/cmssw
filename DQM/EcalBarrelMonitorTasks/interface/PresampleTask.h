#ifndef PresampleTask_H
#define PresampleTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class PresampleTask : public DQWorkerTask {
  public:
    PresampleTask(const edm::ParameterSet &, const edm::ParameterSet&);
    ~PresampleTask();

    bool filterRunType(const std::vector<short>&);

    void analyze(const void*, Collections);

    void runOnDigis(const EcalDigiCollection &);

    enum MESets {
      kPedestal, // profile2d
      nMESets
    };

    static void setMEData(std::vector<MEData>&);
  };

  inline void PresampleTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEBDigi:
    case kEEDigi:
      runOnDigis(*static_cast<const EcalDigiCollection*>(_p));
      break;
    default:
      break;
    }
  }

}

#endif

