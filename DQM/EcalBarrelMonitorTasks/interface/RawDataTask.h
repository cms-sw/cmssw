#ifndef RawDataTask_H
#define RawDataTask_H

#include "DQWorkerTask.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

namespace ecaldqm {

  class RawDataTask : public DQWorkerTask {
  public:
    RawDataTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~RawDataTask() {}

    void setDependencies(DependencySet&);

    void bookMEs();

    void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);
    void beginEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnSource(const FEDRawDataCollection &, Collections);
    void runOnRawData(const EcalRawDataCollection &, Collections);

    enum Constants {
      nEventTypes = 25
    };

  private:
    int hltTaskMode_; // 0 -> Do not produce FED plots; 1 -> Only produce FED plots; 2 -> Do both
    std::string hltTaskFolder_;
    int l1A_;
    int orbit_;
    int bx_;
    short triggerType_;
    int feL1Offset_;

  };

  inline void RawDataTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kSource:
      runOnSource(*static_cast<const FEDRawDataCollection*>(_p), _collection);
      break;
    case kEcalRawData:
      runOnRawData(*static_cast<const EcalRawDataCollection*>(_p), _collection);
      break;
    default:
      break;
    }
  }

}

#endif

