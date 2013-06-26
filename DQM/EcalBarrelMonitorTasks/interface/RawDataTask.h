#ifndef RawDataTask_H
#define RawDataTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

namespace ecaldqm {

  class RawDataTask : public DQWorkerTask {
  public:
    RawDataTask(const edm::ParameterSet &, const edm::ParameterSet &);
    ~RawDataTask();

    void bookMEs();

    void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);
    void beginEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnSource(const FEDRawDataCollection &, Collections);
    void runOnRawData(const EcalRawDataCollection &, Collections);

    enum MESets {
      kEventTypePreCalib, // h1f
      kEventTypeCalib, // h1f
      kEventTypePostCalib, // h1f
      kCRC, // h1f
      kRunNumber, // h1f
      kOrbit, // h1f
      kTriggerType, // h1f
      kL1ADCC, // h1f
      kL1AFE, // h1f
      //      kL1AFEMap, // h2f
      kL1ATCC, // h1f
      kL1ASRP, // h1f
      kBXDCC, // h1f
      kBXFE, // h1f
      kBXTCC, // h1f
      kBXSRP, // h1f
      kDesyncByLumi, // h1f
      kDesyncTotal, // h1f
      kFEStatus, // h1f
      kFEByLumi, // h1f
      kFEDEntries,
      kFEDFatal,
      nMESets
    };

    enum Constants {
      nEventTypes = 25
    };

    static void setMEData(std::vector<MEData>&);

  private:
    int hltTaskMode_; // 0 -> Do not produce FED plots; 1 -> Only produce FED plots; 2 -> Do both
    std::string hltTaskFolder_;
    int run_;
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

