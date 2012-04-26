#ifndef SelectiveReadoutTask_H
#define SelectiveReadoutTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

class EcalTrigTowerConstituentsMap;

namespace ecaldqm {

  class SelectiveReadoutTask : public DQWorkerTask {
  public:
    SelectiveReadoutTask(const edm::ParameterSet &, const edm::ParameterSet &);
    ~SelectiveReadoutTask();

    void beginRun(const edm::Run &, const edm::EventSetup &);
    void beginEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnSource(const FEDRawDataCollection &);
    void runOnRawData(const EcalRawDataCollection &);
    void runOnEBSrFlags(const EBSrFlagCollection &);
    void runOnEESrFlags(const EESrFlagCollection &);
    void runOnDigis(const EcalDigiCollection &, Collections);

    enum MESets {
      kTowerSize, // profile2d
      kDCCSize, // h2f
      kEventSize, // h1f
      kFlagCounterMap, // h2f counter
      kRUForcedMap, // h2f counter
      kFullReadout, // h1f
      kFullReadoutMap, // h2f counter
      kZS1Map, // h2f counter
      kZSMap, // h2f counter
      kZSFullReadout, // h1f
      kZSFullReadoutMap, // h2f counter
      kFRDropped, // h1f
      kFRDroppedMap, // h2f counter
      kHighIntPayload, // h1f
      kLowIntPayload, // h1f
      kHighIntOutput, // h1f
      kLowIntOutput, // h1f
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Constants {
      nFIRTaps = 6,
      bytesPerCrystal = 24
    };

  private:
    void setFIRWeights_(const std::vector<double> &);
    void runOnSrFlag_(const DetId &, int, float&);

    bool useCondDb_;
    int iFirstSample_;
    std::vector<int> ZSFIRWeights_;

    const EcalChannelStatus *channelStatus_;
    const EcalTrigTowerConstituentsMap *ttMap_;
    const EBSrFlagCollection *ebSRFs_;
    const EESrFlagCollection *eeSRFs_;

    std::vector<short> feStatus_[54];
    std::set<uint32_t> frFlaggedTowers_;
    std::set<uint32_t> zsFlaggedTowers_;
    std::map<uint32_t, int> ttCrystals_;

  };

  inline void SelectiveReadoutTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kSource:
      runOnSource(*static_cast<const FEDRawDataCollection*>(_p));
      break;
    case kEcalRawData:
      runOnRawData(*static_cast<const EcalRawDataCollection*>(_p));
      break;
    case kEBSrFlag:
      runOnEBSrFlags(*static_cast<const EBSrFlagCollection*>(_p));
      break;
    case kEESrFlag:
      runOnEESrFlags(*static_cast<const EESrFlagCollection*>(_p));
      break;
    case kEBDigi:
    case kEEDigi:
      runOnDigis(*static_cast<const EcalDigiCollection*>(_p), _collection);
      break;
    default:
      break;
    }
  }

}

#endif

