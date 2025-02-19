#ifndef IntegrityTask_H
#define IntegrityTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

namespace ecaldqm {

  class IntegrityTask : public DQWorkerTask {
  public:
    IntegrityTask(const edm::ParameterSet &, const edm::ParameterSet&);
    ~IntegrityTask();

    void bookMEs();

    void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnErrors(const DetIdCollection &, Collections);
    void runOnErrors(const EcalElectronicsIdCollection &, Collections);

    enum MESets {
      kByLumi,
      kTotal,
      kGain,
      kChId,
      kGainSwitch,
      kBlockSize,
      kTowerId,
      kFEDNonFatal,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

  private:
    int hltTaskMode_; // 0 -> Do not produce FED plots; 1 -> Only produce FED plots; 2 -> Do both
    std::string hltTaskFolder_;
  };

  inline void IntegrityTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kGainErrors:
    case kChIdErrors:
    case kGainSwitchErrors:
      runOnErrors(*static_cast<const DetIdCollection*>(_p), _collection);
      break;
    case kTowerIdErrors:
    case kBlockSizeErrors:
      runOnErrors(*static_cast<const EcalElectronicsIdCollection*>(_p), _collection);
      break;
    default:
      break;
    }
  }

}

#endif

