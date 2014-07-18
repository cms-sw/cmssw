#ifndef RawDataTask_H
#define RawDataTask_H

#include "DQWorkerTask.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "DataFormats/Provenance/interface/RunID.h"

namespace ecaldqm
{
  class RawDataTask : public DQWorkerTask {
  public:
    RawDataTask();
    ~RawDataTask() {}

    void addDependencies(DependencySet&) override;

    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void beginEvent(edm::Event const&, edm::EventSetup const&) override;

    bool analyze(void const*, Collections) override;

    void runOnSource(FEDRawDataCollection const&);
    void runOnRawData(EcalRawDataCollection const&);

    enum Constants {
      nEventTypes = 25
    };

  private:
    edm::RunNumber_t runNumber_; // run number needed regardless of single-/multi-thread operation
    int l1A_;
    int orbit_;
    int bx_;
    short triggerType_;
    int feL1Offset_;

  };

  inline bool RawDataTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kSource:
      if(_p) runOnSource(*static_cast<FEDRawDataCollection const*>(_p));
      return true;
      break;
    case kEcalRawData:
      if(_p) runOnRawData(*static_cast<EcalRawDataCollection const*>(_p));
      return true;
      break;
    default:
      break;
    }
    return false;
  }

}

#endif

