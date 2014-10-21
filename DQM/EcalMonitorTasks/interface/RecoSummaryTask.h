#ifndef RecoSummaryTask_H
#define RecoSummaryTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/View.h"

namespace ecaldqm {

  class RecoSummaryTask : public DQWorkerTask {
  public:
    RecoSummaryTask();
    ~RecoSummaryTask() {}

    bool filterRunType(short const*) override;

    void addDependencies(DependencySet&) override;

    bool analyze(void const*, Collections) override;
    void endEvent(edm::Event const&, edm::EventSetup const&);

    void runOnRecHits(EcalRecHitCollection const&, Collections);
    void runOnReducedRecHits(EcalRecHitCollection const&, Collections);
    void runOnBasicClusters(edm::View<reco::CaloCluster> const&, Collections);

  private:
    void setParams(edm::ParameterSet const&) override;

    float rechitThresholdEB_;
    float rechitThresholdEE_;
    EcalRecHitCollection const* ebHits_;
    EcalRecHitCollection const* eeHits_;
  };

  inline bool RecoSummaryTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kEBRecHit:
    case kEERecHit:
      if(_p) runOnRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
      return true;
      break;
    case kEBReducedRecHit:
    case kEEReducedRecHit:
      if(_p) runOnReducedRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
      return true;
      break;
    case kEBBasicCluster:
    case kEEBasicCluster:
      if(_p) runOnBasicClusters(*static_cast<edm::View<reco::CaloCluster> const*>(_p), _collection);
      return true;
      break;
    default:
      break;
    }
    return false;
  }

}

#endif

