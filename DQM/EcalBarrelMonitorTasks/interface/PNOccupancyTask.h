#ifndef PNOccupancyTask_H
#define PNOccupancyTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class PNOccupancyTask : public DQWorkerTask {
  public:
    PNOccupancyTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PNOccupancyTask() {}

    bool filterRunType(const std::vector<short>&);

    void analyze(const void*, Collections);

    void runOnDigis(const EcalPnDiodeDigiCollection &);

    enum MESets {
      kDigi,
      nMESets
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected:
    bool enable_[BinService::nDCC];
  };

  inline void PNOccupancyTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kPnDiodeDigi:
      runOnDigis(*static_cast<const EcalPnDiodeDigiCollection*>(_p));
      break;
    default:
      break;
    }
  }

}

#endif

