#ifndef SelectiveReadoutTask_H
#define SelectiveReadoutTask_H

#include "DQWorkerTask.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"

class EcalTrigTowerConstituentsMap;

namespace ecaldqm {

  class SelectiveReadoutTask : public DQWorkerTask {
  public:
    SelectiveReadoutTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~SelectiveReadoutTask() {}

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

    static void setMEOrdering(std::map<std::string, unsigned>&);

    enum Constants {
      nFIRTaps = 6,
      bytesPerCrystal = 24,
      nRU = EcalTrigTowerDetId::kEBTotalTowers + EcalScDetId::kSizeForDenseIndexing
    };

  private:
    void setFIRWeights_(const std::vector<double> &);
    void runOnSrFlag_(const DetId &, int, double&);

    bool useCondDb_;
    int iFirstSample_;
    std::vector<int> ZSFIRWeights_;

    std::set<std::pair<int, int> > suppressed_;
    std::vector<short> flags_;
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

