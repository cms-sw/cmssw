#ifndef PNPresampleTask_H
#define PNPresampleTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class PNPresampleTask : public DQWorkerTask {
  public:
    PNPresampleTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PNPresampleTask();

    bool filterRunType(const std::vector<short>&);

    void beginRun(const edm::Run &, const edm::EventSetup &);
    void endEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnPnDigis(const EcalPnDiodeDigiCollection&);

    enum MESets {
      kPedestal,
      nMESets
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

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
