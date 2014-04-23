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
    SelectiveReadoutTask();
    ~SelectiveReadoutTask() {}

    void addDependencies(DependencySet&) override;

    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void beginEvent(edm::Event const&, edm::EventSetup const&) override;

    bool analyze(void const*, Collections) override;

    void runOnSource(FEDRawDataCollection const&);
    void runOnRawData(EcalRawDataCollection const&);
    template<typename SRFlagCollection> void runOnSrFlags(SRFlagCollection const&, Collections);
    template<typename DigiCollection> void runOnDigis(DigiCollection const&, Collections);

    enum Constants {
      nFIRTaps = 6,
      bytesPerCrystal = 24,
      nRU = EcalTrigTowerDetId::kEBTotalTowers + EcalScDetId::kSizeForDenseIndexing
    };

  private:
    void setParams(edm::ParameterSet const&) override;

    void setFIRWeights_(std::vector<double> const&);

    bool useCondDb_;
    int iFirstSample_;
    std::vector<int> ZSFIRWeights_;

    std::set<std::pair<int, int> > suppressed_;
    std::vector<short> flags_;
  };

  inline bool SelectiveReadoutTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kSource:
      if(_p) runOnSource(*static_cast<FEDRawDataCollection const*>(_p));
      return true;
      break;
    case kEcalRawData:
      if(_p) runOnRawData(*static_cast<EcalRawDataCollection const*>(_p));
      return true;
      break;
    case kEBSrFlag:
      if(_p) runOnSrFlags(*static_cast<EBSrFlagCollection const*>(_p), _collection);
      return true;
      break;
    case kEESrFlag:
      if(_p) runOnSrFlags(*static_cast<EESrFlagCollection const*>(_p), _collection);
      return true;
      break;
    case kEBDigi:
      if(_p) runOnDigis(*static_cast<EBDigiCollection const*>(_p), _collection);
      return true;
      break;
    case kEEDigi:
      if(_p) runOnDigis(*static_cast<EEDigiCollection const*>(_p), _collection);
      return true;
      break;
    default:
      break;
    }
    return false;
  }

}

#endif

