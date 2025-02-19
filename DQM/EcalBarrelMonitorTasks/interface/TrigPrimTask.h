#ifndef TrigPrimTask_H
#define TrigPrimTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

class EcalTrigTowerConstituentsMap;

namespace ecaldqm {

  class TrigPrimTask : public DQWorkerTask {
  public:
    TrigPrimTask(const edm::ParameterSet &, const edm::ParameterSet &);
    ~TrigPrimTask();

    void bookMEs();

    void analyze(const void*, Collections);

    void beginRun(const edm::Run &, const edm::EventSetup &);
    void beginEvent(const edm::Event &, const edm::EventSetup &);

    void runOnRealTPs(const EcalTrigPrimDigiCollection &);
    void runOnEmulTPs(const EcalTrigPrimDigiCollection &);
    void runOnDigis(const EcalDigiCollection &);

    enum MESets {
      kEtReal,
      //      kEtEmul,
      kEtMaxEmul,
      kEtRealMap,
      //      kEtEmulMap,
      kEtSummary,
      kMatchedIndex,
      kEmulMaxIndex,
      kTimingError,
      kEtVsBx,
      kOccVsBx,
      kLowIntMap,
      kMedIntMap,
      kHighIntMap,
      kTTFlags,
      kTTFMismatch,
/*       kTimingCalo, */
/*       kTimingMuon, */
      kEtEmulError,
      kFGEmulError,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Constants {
      nBXBins = 15
    };

  private:
    const EcalTrigTowerConstituentsMap* ttMap_;
    const EcalTrigPrimDigiCollection* realTps_;

    bool runOnEmul_;

    int expectedTiming_;
    std::string HLTCaloPath_;
    std::string HLTMuonPath_;
    bool HLTCaloBit_;
    bool HLTMuonBit_;

    int bxBinEdges_[nBXBins + 1];
    float bxBin_;

    std::map<uint32_t, unsigned> towerReadouts_;
  };

  inline void TrigPrimTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kTrigPrimDigi:
      runOnRealTPs(*static_cast<const EcalTrigPrimDigiCollection*>(_p));
      break;
    case kTrigPrimEmulDigi:
      runOnEmulTPs(*static_cast<const EcalTrigPrimDigiCollection*>(_p));
      break;
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

