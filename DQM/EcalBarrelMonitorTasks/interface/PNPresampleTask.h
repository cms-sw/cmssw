#ifndef PNPresampleTask_H
#define PNPresampleTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class PNPresampleTask : public DQWorkerTask {
  public:
    PNPresampleTask(const edm::ParameterSet &, const edm::ParameterSet &);
    ~PNPresampleTask();

    bool filterRunType(const std::vector<short>&) override;

    void beginRun(const edm::Run &, const edm::EventSetup &) override;
    void endEvent(const edm::Event &, const edm::EventSetup &) override;

    void analyze(const void*, Collections) override;

    void runOnPnDigis(const EcalPnDiodeDigiCollection&);

    enum MESets {
      kPedestal,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

  protected:
    bool enable_[BinService::nDCC];
  };

  inline void PNPresampleTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kPnDiodeDigi:
      runOnPnDigis(*static_cast<const EcalPnDiodeDigiCollection*>(_p));
      break;
    default:
      break;
    }
  }

}

#endif
