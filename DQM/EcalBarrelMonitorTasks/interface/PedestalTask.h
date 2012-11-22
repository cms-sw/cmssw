#ifndef PedestalTask_H
#define PedestalTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class PedestalTask : public DQWorkerTask {
  public:
    PedestalTask(const edm::ParameterSet &, const edm::ParameterSet &);
    ~PedestalTask();

    void bookMEs();

    bool filterRunType(const std::vector<short>&);

    void analyze(const void*, Collections);

    void runOnDigis(const EcalDigiCollection&);
    void runOnPnDigis(const EcalPnDiodeDigiCollection&);

    enum Constants {
      nGain = 3,
      nPNGain = 2
    };

    enum MESets {
      kOccupancy, // h2f
      kPedestal = kOccupancy + nGain, // profile2d
      kPNOccupancy = kPedestal + nGain,
      kPNPedestal = kPNOccupancy + nPNGain, // profile2d
      nMESets = kPNPedestal + nPNGain
    };

    static void setMEData(std::vector<MEData>&);

  protected:
    std::vector<int> MGPAGains_;
    std::vector<int> MGPAGainsPN_;

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
