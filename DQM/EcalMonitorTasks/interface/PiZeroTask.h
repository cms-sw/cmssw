#ifndef DQM_EcalMonitorTasks_PiZeroTask_H
#define DQM_EcalMonitorTasks_PiZeroTask_H

#include "DQM/EcalMonitorTasks/interface/DQWorkerTask.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

#include "TVector3.h"

namespace ecaldqm {

  class PiZeroTask : public DQWorkerTask {
  public:
    PiZeroTask();
    ~PiZeroTask() override = default;

    bool filterRunType(short const*) override;
    bool analyze(void const*, Collections) override;
    void runOnEBRecHits(EcalRecHitCollection const&);

  private:
    void setParams(edm::ParameterSet const&) override;

    static const int MAXCLUS = 2000;
    static const int MAXPI0S = 200;

    // Parameters needed for pi0 finding
    double seleXtalMinEnergy_;

    double clusSeedThr_;
    int clusEtaSize_;
    int clusPhiSize_;

    double selePtGammaOne_;
    double selePtGammaTwo_;
    double seleS4S9GammaOne_;
    double seleS4S9GammaTwo_;
    double selePtPi0_;
    double selePi0Iso_;
    double selePi0BeltDR_;
    double selePi0BeltDeta_;
    double seleMinvMaxPi0_;
    double seleMinvMinPi0_;

    edm::ParameterSet posCalcParameters_;
  };

  inline bool PiZeroTask::analyze(void const* collection_data, Collections collection) {
    switch (collection) {
      case kEBRecHit:
        if (collection_data)
          runOnEBRecHits(*static_cast<EcalRecHitCollection const*>(collection_data));
        return true;
      case kEERecHit:  // This module does not run on EERecHits
        break;
      default:
        break;
    }

    return false;
  }

}  // namespace ecaldqm

#endif
