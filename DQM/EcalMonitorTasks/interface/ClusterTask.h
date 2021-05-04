#ifndef ClusterTask_H
#define ClusterTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <bitset>

namespace ecaldqm {
  class ClusterTask : public DQWorkerTask {
  public:
    ClusterTask();
    ~ClusterTask() override {}

    bool filterRunType(short const*) override;

    void addDependencies(DependencySet&) override;

    void beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) override;
    void endEvent(edm::Event const&, edm::EventSetup const&) override;

    bool analyze(void const*, Collections) override;

    void runOnRecHits(EcalRecHitCollection const&, Collections);
    void runOnBasicClusters(edm::View<reco::CaloCluster> const&, Collections);
    void runOnSuperClusters(reco::SuperClusterCollection const&, Collections);

    void setTokens(edm::ConsumesCollector&) override;

    enum TriggerTypes { kEcalTrigger, kHcalTrigger, kCSCTrigger, kDTTrigger, kRPCTrigger, nTriggerTypes };

  private:
    void setParams(edm::ParameterSet const&) override;

    EcalRecHitCollection const* ebHits_;
    EcalRecHitCollection const* eeHits_;
    //    int ievt_;
    //    int massCalcPrescale_;
    bool doExtra_;
    float energyThreshold_;
    float swissCrossMaxThreshold_;
    std::vector<std::string> egTriggerAlgos_;
    std::bitset<nTriggerTypes> triggered_;
    unsigned trigTypeToME_[nTriggerTypes];

    edm::InputTag L1GlobalTriggerReadoutRecordTag_;
    edm::InputTag L1MuGMTReadoutCollectionTag_;
    edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> L1GlobalTriggerReadoutRecordToken_;
    edm::EDGetTokenT<L1MuGMTReadoutCollection> L1MuGMTReadoutCollectionToken_;
  };

  inline bool ClusterTask::analyze(void const* _p, Collections _collection) {
    switch (_collection) {
      case kEBRecHit:
      case kEERecHit:
        if (_p)
          runOnRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
        return true;
        break;
      case kEBBasicCluster:
      case kEEBasicCluster:
        if (_p)
          runOnBasicClusters(*static_cast<edm::View<reco::CaloCluster> const*>(_p), _collection);
        return true;
        break;
      case kEBSuperCluster:
      case kEESuperCluster:
        if (_p)
          runOnSuperClusters(*static_cast<reco::SuperClusterCollection const*>(_p), _collection);
        return true;
        break;
      default:
        break;
    }

    return false;
  }

}  // namespace ecaldqm

#endif
