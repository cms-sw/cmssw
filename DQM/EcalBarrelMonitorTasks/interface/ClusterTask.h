#ifndef ClusterTask_H
#define ClusterTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

class CaloTopology;
class CaloSubdetectorGeometry;

namespace ecaldqm {

  class ClusterTask : public DQWorkerTask {
  public:
    ClusterTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~ClusterTask() {}

    bool filterRunType(const std::vector<short>&);

    void setDependencies(DependencySet&);

    void beginRun(const edm::Run &, const edm::EventSetup &);
    void beginEvent(const edm::Event &, const edm::EventSetup &);

    void analyze(const void*, Collections);

    void runOnRecHits(const EcalRecHitCollection &, Collections);
    void runOnBasicClusters(const reco::BasicClusterCollection &, Collections);
    void runOnSuperClusters(const reco::SuperClusterCollection &, Collections);

  private:
    const CaloTopology *topology_;
    const CaloSubdetectorGeometry* ebGeometry_;
    const CaloSubdetectorGeometry* eeGeometry_;
    const EcalRecHitCollection *ebHits_, *eeHits_;
    int ievt_;
    //    int massCalcPrescale_;
  };

  inline void ClusterTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kEBRecHit:
    case kEERecHit:
      runOnRecHits(*static_cast<const EcalRecHitCollection*>(_p), _collection);
      break;
    case kEBBasicCluster:
    case kEEBasicCluster:
      runOnBasicClusters(*static_cast<const reco::BasicClusterCollection*>(_p), _collection);
      break;
    case kEBSuperCluster:
    case kEESuperCluster:
      runOnSuperClusters(*static_cast<const reco::SuperClusterCollection*>(_p), _collection);
      break;
    default:
      break;
    }
  }

}

#endif

