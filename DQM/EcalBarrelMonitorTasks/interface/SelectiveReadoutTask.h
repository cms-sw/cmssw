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

    void setDependencies(DependencySet&);

    void beginRun(const edm::Run &, const edm::EventSetup &);
    void beginEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnSource(const FEDRawDataCollection &);
    void runOnRawData(const EcalRawDataCollection &);
    void runOnEBSrFlags(const EBSrFlagCollection &);
    void runOnEESrFlags(const EESrFlagCollection &);
    void runOnDigis(const EcalDigiCollection &, Collections);

    enum Constants {
      nFIRTaps = 6,
      bytesPerCrystal = 24,
      nRU = EcalTrigTowerDetId::kEBTotalTowers + EcalScDetId::kSizeForDenseIndexing
    };

  private:
    void setFIRWeights_(const std::vector<double> &);
    void runOnSrFlag_(const DetId &, int, double&, MESet**);

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

